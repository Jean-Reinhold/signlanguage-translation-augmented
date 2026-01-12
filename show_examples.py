import pandas as pd
import os

# Load the partial data from the last test run
csv_path = '/mnt/disk3Tb/augmented-slt-datasets/logs/RWTH_PHOENIX_2014T_20260111_185554/partial_data.csv'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    
    # Filter for augmented rows
    variants = df[df['augmentation_method'] == 'back_translate']
    
    print("\n" + "="*100)
    print(f"{'ORIGINAL TEXT':<45} | {'PIVOT':<5} | {'AUGMENTED TEXT (BACK-TRANSLATION)':<45}")
    print("-" * 100)
    
    for _, row in variants.head(10).iterrows():
        # Find the original text for this ID
        orig_row = df[(df['id'] == row['id']) & (df['augmentation_method'].isna())]
        if not orig_row.empty:
            orig_text = orig_row.iloc[0]['text']
            # Truncate for display if needed
            orig_disp = (orig_text[:42] + '...') if len(orig_text) > 42 else orig_text
            aug_disp = (row['text'][:42] + '...') if len(row['text']) > 42 else row['text']
            
            print(f"{orig_disp:<45} | {row['augmentation_pivot']:<5} | {aug_disp:<45}")
    
    print("="*100 + "\n")
else:
    print(f"Error: File not found at {csv_path}")
