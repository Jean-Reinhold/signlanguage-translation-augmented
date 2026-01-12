#!/bin/bash
#
# Run text augmentation for all SLT datasets using back-translation.
#
# This script processes each dataset sequentially with proper progress tracking
# and generates comprehensive statistics for each run.
#
# Usage:
#   ./scripts/run_augmentation.sh                    # Run all datasets
#   ./scripts/run_augmentation.sh --dataset GSL      # Run single dataset
#   ./scripts/run_augmentation.sh --resume           # Resume interrupted runs
#   ./scripts/run_augmentation.sh --dry-run          # Show what would be done
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
    echo -e "${GREEN}✓ Loaded .env file${NC}"
else
    echo -e "${YELLOW}⚠ No .env file found at $PROJECT_DIR/.env${NC}"
fi

# Default values
DATASETS=("RWTH_PHOENIX_2014T" "lsat" "How2Sign" "ISL" "LSFB-CONT" "GSL")
SPLIT="train"
BATCH_SIZE=8
NUM_VARIANTS=2
LOG_LEVEL="INFO"
RESUME=false
DRY_RUN=false
PREDICT_ONLY=false
SINGLE_DATASET=""

# Set up python command (use venv if available)
PYTHON_CMD="python3"
if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON_CMD="$PROJECT_DIR/.venv/bin/python"
fi

# Dataset sample counts (approximate for train split)
declare -A DATASET_SAMPLES
DATASET_SAMPLES["RWTH_PHOENIX_2014T"]=7096
DATASET_SAMPLES["lsat"]=14880
DATASET_SAMPLES["How2Sign"]=35191
DATASET_SAMPLES["ISL"]=31222
DATASET_SAMPLES["LSFB-CONT"]=27500
DATASET_SAMPLES["GSL"]=40826

# Dataset languages and pivot languages
declare -A DATASET_LANG
DATASET_LANG["RWTH_PHOENIX_2014T"]="German → English pivot"
DATASET_LANG["lsat"]="Spanish → English pivot"
DATASET_LANG["How2Sign"]="English → Spanish pivot"
DATASET_LANG["ISL"]="English → Spanish pivot"
DATASET_LANG["LSFB-CONT"]="French → English pivot"
DATASET_LANG["GSL"]="Greek → English pivot"

# Number of pivots per dataset (all use single pivot now)
declare -A DATASET_PIVOTS
DATASET_PIVOTS["RWTH_PHOENIX_2014T"]=1
DATASET_PIVOTS["lsat"]=1
DATASET_PIVOTS["How2Sign"]=1
DATASET_PIVOTS["ISL"]=1
DATASET_PIVOTS["LSFB-CONT"]=1
DATASET_PIVOTS["GSL"]=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            SINGLE_DATASET="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-variants)
            NUM_VARIANTS="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --predict)
            PREDICT_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset NAME     Run only specified dataset"
            echo "  --split NAME       Dataset split (train/val/test, default: train)"
            echo "  --batch-size N     Batch size for API calls (default: 8)"
            echo "  --num-variants N   Number of variants per pivot language (default: 2)"
            echo "  --log-level LEVEL  Logging level (default: INFO)"
            echo "  --resume           Resume from checkpoint if available"
            echo "  --dry-run          Show what would be done without executing"
            echo "  --predict          Predict costs and stats before running"
            echo "  --help             Show this help message"
            echo ""
            echo "Available datasets:"
            for ds in "${DATASETS[@]}"; do
                samples=${DATASET_SAMPLES[$ds]:-0}
                pivots=${DATASET_PIVOTS[$ds]:-1}
                echo "  - $ds ($samples samples, $pivots pivot(s))"
            done
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# If single dataset specified, use only that
if [ -n "$SINGLE_DATASET" ]; then
    DATASETS=("$SINGLE_DATASET")
fi

# Calculate total datasets
TOTAL_DATASETS=${#DATASETS[@]}
CURRENT=0
SUCCESSFUL=0
FAILED=0
SKIPPED=0

# Results array
declare -A RESULTS

# Print header
print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}║          SLT DATASET AUGMENTATION WITH BACK-TRANSLATION            ║${NC}"
    echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Print configuration
print_config() {
    echo -e "${CYAN}Configuration:${NC}"
    echo -e "  ${BOLD}Azure Endpoint:${NC}    ${AZURE_OPENAI_ENDPOINT:-not set}"
    echo -e "  ${BOLD}Azure Deployment:${NC}  ${AZURE_OPENAI_DEPLOYMENT:-not set}"
    echo -e "  ${BOLD}API Version:${NC}       ${AZURE_OPENAI_API_VERSION:-not set}"
    echo -e "  ${BOLD}Datasets Dir:${NC}      ${SLT_DATASETS_DIR:-/mnt/disk3Tb/slt-datasets}"
    echo -e "  ${BOLD}Output Dir:${NC}        ${AUGMENTED_DATASETS_DIR:-/mnt/disk3Tb/augmented-slt-datasets}"
    echo -e "  ${BOLD}Split:${NC}             $SPLIT"
    echo -e "  ${BOLD}Batch Size:${NC}        $BATCH_SIZE"
    echo -e "  ${BOLD}Num Variants:${NC}      $NUM_VARIANTS (variants per pivot language)"
    echo -e "  ${BOLD}Resume:${NC}            $RESUME"
    echo ""
    
    # Calculate totals
    local total_samples=0
    local total_augmented=0
    local total_api_calls=0
    
    echo -e "${CYAN}Datasets to process:${NC}"
    echo ""
    printf "  ${BOLD}%-22s %10s %8s %12s %12s  %-28s${NC}\n" "Dataset" "Samples" "Pivots" "Variants" "Total" "Language"
    echo "  ─────────────────────────────────────────────────────────────────────────────────────────────────"
    
    for ds in "${DATASETS[@]}"; do
        local samples=${DATASET_SAMPLES[$ds]:-0}
        local pivots=${DATASET_PIVOTS[$ds]:-1}
        local variants=$((samples * pivots * AUG_FACTOR))
        local augmented=$((samples + variants))
        local lang=${DATASET_LANG[$ds]:-"Unknown"}
        
        total_samples=$((total_samples + samples))
        total_augmented=$((total_augmented + augmented))
        total_api_calls=$((total_api_calls + samples * pivots * 2))  # 2 API calls per pivot (forward + back)
        
        printf "  %-22s %10s %8s %12s %12s  %-28s\n" "$ds" "$samples" "$pivots" "~$variants" "~$augmented" "$lang"
    done
    
    echo "  ─────────────────────────────────────────────────────────────────────────────────────────────────"
    local total_variants=$((total_augmented - total_samples))
    printf "  ${BOLD}%-22s %10s %8s %12s %12s${NC}\n" "TOTAL" "$total_samples" "-" "~$total_variants" "~$total_augmented"
    echo ""
    
    # Estimate time and cost using the python script's prediction logic
    echo -e "  ${YELLOW}Calculating detailed cost prediction...${NC}"
    local DS_ARG=""
    if [ -n "$SINGLE_DATASET" ]; then
        DS_ARG="--dataset $SINGLE_DATASET"
    else
        DS_ARG="--all-datasets"
    fi
    $PYTHON_CMD -m src.augmentation.augment_dataset $DS_ARG --split $SPLIT --num-variants $NUM_VARIANTS --predict-only --quiet
    
    echo ""
}

# Print progress bar
print_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))
    
    # Use a fixed line for progress at the top or bottom
    # For now, just ensure it doesn't use \r if we are about to print other things
    echo -e "${CYAN}Overall Progress: ${NC}[" | tr -d '\n'
    printf "%${filled}s" | tr ' ' '█' | tr -d '\n'
    printf "%${empty}s" | tr ' ' '░' | tr -d '\n'
    printf "] ${BOLD}%3d%%${NC} (%d/%d datasets)\n" $percentage $current $total
}

# Print dataset header
print_dataset_header() {
    local dataset=$1
    local index=$2
    local total=$3
    
    echo ""
    echo -e "${BOLD}${YELLOW}┌──────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BOLD}${YELLOW}│ Dataset $index/$total: $dataset${NC}"
    echo -e "${BOLD}${YELLOW}└──────────────────────────────────────────────────────────────────┘${NC}"
}

# Run augmentation for a single dataset
run_augmentation() {
    local dataset=$1
    local start_time=$(date +%s)
    
    # Get dataset info for display
    local samples=${DATASET_SAMPLES[$dataset]:-0}
    local pivots=${DATASET_PIVOTS[$dataset]:-1}
    local expected_variants=$((samples * pivots * AUG_FACTOR))
    local expected_total=$((samples + expected_variants))
    
    echo -e "${CYAN}Expected: $samples samples → ~$expected_total total ($expected_variants new variants)${NC}"
    echo ""
    
    # Build command
    local cmd="$PYTHON_CMD -m src.augmentation.augment_dataset"
    cmd+=" --dataset $dataset"
    cmd+=" --split $SPLIT"
    cmd+=" --batch-size $BATCH_SIZE"
    cmd+=" --num-variants $NUM_VARIANTS"
    cmd+=" --log-level $LOG_LEVEL"
    
    if [ "$RESUME" = true ]; then
        cmd+=" --resume"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] Would execute:${NC}"
        echo "  $cmd"
        return 0
    fi
    
    echo -e "${CYAN}Executing:${NC} $cmd"
    echo ""
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    # Run the command
    if $cmd; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        
        echo ""
        echo -e "${GREEN}✓ $dataset completed successfully in ${minutes}m ${seconds}s${NC}"
        RESULTS[$dataset]="SUCCESS (${minutes}m ${seconds}s)"
        return 0
    else
        echo ""
        echo -e "${RED}✗ $dataset failed${NC}"
        RESULTS[$dataset]="FAILED"
        return 1
    fi
}

# Print final summary
print_summary() {
    echo ""
    echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}║                        AUGMENTATION SUMMARY                        ║${NC}"
    echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    printf "%-25s %s\n" "Dataset" "Status"
    echo "────────────────────────────────────────────────────"
    
    for dataset in "${DATASETS[@]}"; do
        local status="${RESULTS[$dataset]:-NOT RUN}"
        if [[ "$status" == SUCCESS* ]]; then
            printf "%-25s ${GREEN}%s${NC}\n" "$dataset" "$status"
        elif [[ "$status" == "FAILED" ]]; then
            printf "%-25s ${RED}%s${NC}\n" "$dataset" "$status"
        else
            printf "%-25s ${YELLOW}%s${NC}\n" "$dataset" "$status"
        fi
    done
    
    echo "────────────────────────────────────────────────────"
    echo -e "${GREEN}Successful:${NC} $SUCCESSFUL  ${RED}Failed:${NC} $FAILED  ${YELLOW}Skipped:${NC} $SKIPPED"
    echo ""
    
    # Print output directory info
    local output_dir="${AUGMENTED_DATASETS_DIR:-/mnt/disk3Tb/augmented-slt-datasets}"
    echo -e "${CYAN}Output files saved to:${NC} $output_dir"
    echo ""
    
    if [ $FAILED -gt 0 ]; then
        echo -e "${YELLOW}To retry failed datasets, run with --resume flag${NC}"
    fi
}

# Main execution
main() {
    print_header
    print_config
    
    # Handle prediction only
    if [ "$PREDICT_ONLY" = true ]; then
        echo -e "${BOLD}${BLUE}=== RUNNING COST PREDICTION ===${NC}"
        
        local DS_ARG=""
        if [ -n "$SINGLE_DATASET" ]; then
            DS_ARG="--dataset $SINGLE_DATASET"
        else
            DS_ARG="--all-datasets"
        fi
        
        # Call Python script with predict-only
        $PYTHON_CMD -m src.augmentation.augment_dataset $DS_ARG --split $SPLIT --num-variants $NUM_VARIANTS --predict-only
        exit 0
    fi
    
    # Validate environment
    if [ -z "$AZURE_OPENAI_ENDPOINT" ] || [ -z "$AZURE_OPENAI_API_KEY" ]; then
        echo -e "${RED}Error: Azure OpenAI credentials not set${NC}"
        echo "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env file"
        exit 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}=== DRY RUN MODE - No changes will be made ===${NC}"
        echo ""
    fi
    
    # Process each dataset
    for dataset in "${DATASETS[@]}"; do
        CURRENT=$((CURRENT + 1))
        print_progress $CURRENT $TOTAL_DATASETS
        print_dataset_header "$dataset" $CURRENT $TOTAL_DATASETS
        
        if run_augmentation "$dataset"; then
            SUCCESSFUL=$((SUCCESSFUL + 1))
        else
            FAILED=$((FAILED + 1))
        fi
        
        # Update progress
        print_progress $CURRENT $TOTAL_DATASETS
    done
    
    echo ""  # New line after progress bar
    print_summary
    
    # Exit with appropriate code
    if [ $FAILED -gt 0 ]; then
        exit 1
    fi
}

# Run main
main
