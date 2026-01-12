"""
Back-Translation Module for Text Augmentation.

This module implements back-translation augmentation:
1. Translate source text to pivot language(s)
2. Translate back to source language
3. Filter variants by semantic similarity
4. Track statistics and detect duplicates

The back-translation process naturally introduces paraphrases
while preserving meaning, which is ideal for training data augmentation.
"""

import asyncio
import hashlib
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from openai import AsyncAzureOpenAI, AsyncOpenAI

from .prompts import (
    PIVOT_LANGUAGES,
    get_batched_back_translation_prompt,
    get_dataset_prompts,
    get_source_language,
    get_translation_prompts,
    TranslationPrompt,
)
from .rate_limiter import AzureRateLimiter

import json

logger = logging.getLogger(__name__)


# ============================================================================
# Similarity Metrics
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    return text


def jaccard_similarity(a: str, b: str) -> float:
    """
    Compute Jaccard similarity between two strings based on character sets.
    
    Returns value between 0 (completely different) and 1 (identical).
    """
    a_chars = set(normalize_text(a))
    b_chars = set(normalize_text(b))
    if not a_chars or not b_chars:
        return 0.0
    intersection = len(a_chars & b_chars)
    union = len(a_chars | b_chars)
    return intersection / union if union > 0 else 0.0


def word_overlap_similarity(a: str, b: str) -> float:
    """
    Compute word overlap similarity (Jaccard on words).
    
    Better for detecting semantic similarity than character-level.
    """
    a_words = set(normalize_text(a).split())
    b_words = set(normalize_text(b).split())
    if not a_words or not b_words:
        return 0.0
    intersection = len(a_words & b_words)
    union = len(a_words | b_words)
    return intersection / union if union > 0 else 0.0


def levenshtein_similarity(a: str, b: str) -> float:
    """
    Compute normalized Levenshtein similarity.
    
    Returns 1 - (edit_distance / max_length), so 1 = identical, 0 = completely different.
    """
    a = normalize_text(a)
    b = normalize_text(b)
    
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    
    # Simple Levenshtein implementation
    m, n = len(a), len(b)
    if m > n:
        a, b = b, a
        m, n = n, m
    
    current = list(range(m + 1))
    for i in range(1, n + 1):
        previous, current = current, [i] + [0] * m
        for j in range(1, m + 1):
            add, delete, change = previous[j] + 1, current[j - 1] + 1, previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current[j] = min(add, delete, change)
    
    max_len = max(len(a), len(b))
    return 1.0 - (current[m] / max_len) if max_len > 0 else 1.0


def ngram_similarity(a: str, b: str, n: int = 3) -> float:
    """
    Compute n-gram (character) similarity.
    
    Good for detecting near-duplicates and minor variations.
    """
    def get_ngrams(text: str, n: int) -> Counter:
        text = normalize_text(text)
        return Counter(text[i:i+n] for i in range(len(text) - n + 1))
    
    a_ngrams = get_ngrams(a, n)
    b_ngrams = get_ngrams(b, n)
    
    if not a_ngrams or not b_ngrams:
        return 0.0
    
    intersection = sum((a_ngrams & b_ngrams).values())
    union = sum((a_ngrams | b_ngrams).values())
    
    return intersection / union if union > 0 else 0.0


@dataclass
class SimilarityScores:
    """Collection of similarity metrics between two texts."""
    jaccard_char: float = 0.0
    jaccard_word: float = 0.0
    levenshtein: float = 0.0
    ngram_3: float = 0.0
    is_exact_duplicate: bool = False
    is_near_duplicate: bool = False
    
    @property
    def average(self) -> float:
        """Average of all similarity scores."""
        return (self.jaccard_char + self.jaccard_word + 
                self.levenshtein + self.ngram_3) / 4
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "jaccard_char": round(self.jaccard_char, 4),
            "jaccard_word": round(self.jaccard_word, 4),
            "levenshtein": round(self.levenshtein, 4),
            "ngram_3": round(self.ngram_3, 4),
            "average": round(self.average, 4),
            "is_exact_duplicate": self.is_exact_duplicate,
            "is_near_duplicate": self.is_near_duplicate,
        }


def compute_similarity(a: str, b: str, near_duplicate_threshold: float = 0.95) -> SimilarityScores:
    """
    Compute all similarity metrics between two strings.
    
    Args:
        a: First string
        b: Second string
        near_duplicate_threshold: Threshold for near-duplicate detection
    
    Returns:
        SimilarityScores object with all metrics
    """
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    
    is_exact = a_norm == b_norm
    
    scores = SimilarityScores(
        jaccard_char=jaccard_similarity(a, b),
        jaccard_word=word_overlap_similarity(a, b),
        levenshtein=levenshtein_similarity(a, b),
        ngram_3=ngram_similarity(a, b, n=3),
        is_exact_duplicate=is_exact,
    )
    
    # Near-duplicate if average similarity is very high
    scores.is_near_duplicate = scores.average >= near_duplicate_threshold
    
    return scores


# ============================================================================
# Statistics Tracking
# ============================================================================

@dataclass
class AugmentationStatistics:
    """Statistics for an augmentation run."""
    total_originals: int = 0
    total_variants_generated: int = 0
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    variants_kept: int = 0
    failed_translations: int = 0
    
    # Similarity score distributions
    similarity_scores: List[float] = field(default_factory=list)
    
    # Per-pivot statistics
    pivot_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Duplicate tracking
    seen_hashes: Set[str] = field(default_factory=set)
    duplicate_texts: List[str] = field(default_factory=list)
    
    def record_variant(
        self,
        original: str,
        variant: str,
        pivot: str,
        similarity: SimilarityScores,
    ) -> bool:
        """
        Record a generated variant and check for duplicates.
        
        Returns:
            True if variant is unique and should be kept, False if duplicate
        """
        self.total_variants_generated += 1
        
        # Initialize pivot stats
        if pivot not in self.pivot_stats:
            self.pivot_stats[pivot] = {
                "generated": 0,
                "kept": 0,
                "exact_duplicates": 0,
                "near_duplicates": 0,
            }
        self.pivot_stats[pivot]["generated"] += 1
        
        # Record similarity
        self.similarity_scores.append(similarity.average)
        
        # Check for exact duplicate
        variant_hash = hashlib.md5(normalize_text(variant).encode()).hexdigest()
        
        if similarity.is_exact_duplicate:
            self.exact_duplicates_removed += 1
            self.pivot_stats[pivot]["exact_duplicates"] += 1
            return False
        
        if variant_hash in self.seen_hashes:
            self.exact_duplicates_removed += 1
            self.pivot_stats[pivot]["exact_duplicates"] += 1
            self.duplicate_texts.append(variant[:50])
            return False
        
        if similarity.is_near_duplicate:
            self.near_duplicates_removed += 1
            self.pivot_stats[pivot]["near_duplicates"] += 1
            return False
        
        # Variant is unique
        self.seen_hashes.add(variant_hash)
        self.variants_kept += 1
        self.pivot_stats[pivot]["kept"] += 1
        return True
    
    def to_dict(self) -> Dict:
        """Convert statistics to dictionary for logging/JSON."""
        import statistics as stats_module
        
        sim_stats = {}
        if self.similarity_scores:
            sim_stats = {
                "min": round(min(self.similarity_scores), 4),
                "max": round(max(self.similarity_scores), 4),
                "mean": round(stats_module.mean(self.similarity_scores), 4),
                "median": round(stats_module.median(self.similarity_scores), 4),
                "stdev": round(stats_module.stdev(self.similarity_scores), 4) if len(self.similarity_scores) > 1 else 0,
            }
        
        return {
            "total_originals": self.total_originals,
            "total_variants_generated": self.total_variants_generated,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "near_duplicates_removed": self.near_duplicates_removed,
            "variants_kept": self.variants_kept,
            "failed_translations": self.failed_translations,
            "expansion_ratio": round(
                (self.total_originals + self.variants_kept) / max(self.total_originals, 1), 2
            ),
            "duplicate_rate": round(
                (self.exact_duplicates_removed + self.near_duplicates_removed) / 
                max(self.total_variants_generated, 1), 4
            ),
            "similarity_distribution": sim_stats,
            "per_pivot_stats": self.pivot_stats,
        }
    
    def print_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "AUGMENTATION STATISTICS",
            "=" * 60,
            f"  Original samples:      {self.total_originals:,}",
            f"  Variants generated:    {self.total_variants_generated:,}",
            f"  Exact duplicates:      {self.exact_duplicates_removed:,}",
            f"  Near duplicates:       {self.near_duplicates_removed:,}",
            f"  Variants kept:         {self.variants_kept:,}",
            f"  Failed translations:   {self.failed_translations:,}",
            f"  Expansion ratio:       {(self.total_originals + self.variants_kept) / max(self.total_originals, 1):.2f}x",
            "-" * 60,
        ]
        
        if self.similarity_scores:
            import statistics as stats_module
            lines.extend([
                "SIMILARITY SCORES (to original):",
                f"  Min:    {min(self.similarity_scores):.4f}",
                f"  Max:    {max(self.similarity_scores):.4f}",
                f"  Mean:   {stats_module.mean(self.similarity_scores):.4f}",
                f"  Median: {stats_module.median(self.similarity_scores):.4f}",
                "-" * 60,
            ])
        
        if self.pivot_stats:
            lines.append("PER-PIVOT STATISTICS:")
            for pivot, pstats in self.pivot_stats.items():
                lines.append(
                    f"  {pivot}: generated={pstats['generated']}, "
                    f"kept={pstats['kept']}, "
                    f"exact_dup={pstats['exact_duplicates']}, "
                    f"near_dup={pstats['near_duplicates']}"
                )
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# Translation Result with Similarity Scores
# ============================================================================

@dataclass
class TranslationResult:
    """Result of a back-translation operation with similarity scores."""
    original: str
    pivot_translations: Dict[str, str]  # pivot_lang -> translation
    back_translations: Dict[str, List[str]]   # pivot_lang -> list of back_translations
    similarity_scores: Dict[str, List[SimilarityScores]] = field(default_factory=dict)  # pivot -> list of scores
    is_duplicate: Dict[str, List[bool]] = field(default_factory=dict)  # pivot -> list of is_dup flags
    
    @property
    def variants(self) -> List[str]:
        """Get all back-translated variants (excluding duplicates)."""
        result = []
        for pivot, variants in self.back_translations.items():
            dup_flags = self.is_duplicate.get(pivot, [False] * len(variants))
            for i, v in enumerate(variants):
                if i < len(dup_flags) and not dup_flags[i]:
                    result.append(v)
        return result
    
    @property
    def all_texts(self) -> List[str]:
        """Get original plus all non-duplicate variants."""
        return [self.original] + self.variants
    
    @property
    def all_variants_with_scores(self) -> List[Tuple[str, int, str, SimilarityScores]]:
        """Get all variants with their pivot language, index, and scores."""
        result = []
        for pivot, variants in self.back_translations.items():
            scores_list = self.similarity_scores.get(pivot, [])
            for i, text in enumerate(variants):
                score = scores_list[i] if i < len(scores_list) else SimilarityScores()
                result.append((pivot, i, text, score))
        return result
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "original": self.original,
            "pivot_translations": self.pivot_translations,
            "back_translations": self.back_translations,
            "similarity_scores": {
                pivot: [s.to_dict() for s in scores_list]
                for pivot, scores_list in self.similarity_scores.items()
            },
            "is_duplicate": self.is_duplicate,
            "variants_kept": len(self.variants),
        }


class BackTranslator:
    """
    Back-translation augmentation using Azure OpenAI.
    
    Implements the pivot language strategy from agents.md:
    - For non-EN/DE targets → Two pivots: t→EN→t and t→DE→t
    - For EN/DE targets → Spanish pivot
    
    Features:
    - Multiple similarity metrics (Jaccard, Levenshtein, n-gram)
    - Duplicate detection (exact and near-duplicates)
    - Comprehensive statistics tracking
    """
    
    def __init__(
        self,
        source_lang: str,
        pivot_langs: Optional[Tuple[str, ...]] = None,
        rate_limiter: Optional[AzureRateLimiter] = None,
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
        batch_size: int = 8,
        num_variants: int = 2,
        min_similarity: float = 0.3,
        max_similarity: float = 0.95,
        deduplicate: bool = True,
    ):
        """
        Initialize the back-translator.
        
        Args:
            source_lang: Source language code (e.g., 'de', 'en')
            pivot_langs: Pivot language codes; auto-detected if None
            rate_limiter: Rate limiter instance; created if None
            model: Azure OpenAI deployment name (default: $AZURE_OPENAI_DEPLOYMENT)
            temperature: Sampling temperature (must be 1.0 for reasoning models)
            max_tokens: Maximum tokens per response
            batch_size: Number of concurrent API requests
            num_variants: Number of back-translation variants per pivot (default: 2)
            min_similarity: Minimum similarity to keep (filter too different)
            max_similarity: Maximum similarity before marking as duplicate
            deduplicate: Whether to remove duplicates
        """
        self.source_lang = source_lang
        self.pivot_langs = pivot_langs or PIVOT_LANGUAGES.get(source_lang, ("en",))
        self.rate_limiter = rate_limiter or AzureRateLimiter()
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.num_variants = num_variants
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
        self.deduplicate = deduplicate
        
        # Use deployment name from environment variable if not explicitly specified
        self.model = model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
        
        # Reasoning models (like o1/gpt-5-mini/gpt-4.1-mini) require temperature=1.0 and more tokens
        # because max_completion_tokens includes reasoning tokens.
        self.is_reasoning_model = "mini" in self.model.lower() or "o1" in self.model.lower()
        
        if self.is_reasoning_model:
            self.temperature = 1.0
            if temperature != 1.0:
                logger.info(f"Forcing temperature to 1.0 for reasoning model {self.model}")
            
            # Increase max_tokens for reasoning models if not explicitly set high
            if max_tokens < 2048:
                self.max_tokens = 2048
                logger.info(f"Increasing max_completion_tokens to {self.max_tokens} for reasoning model {self.model}")
            else:
                self.max_tokens = max_tokens
        else:
            self.temperature = temperature
            self.max_tokens = max_tokens
        
        # Build batched prompts for all pivot languages
        self.batched_prompts: Dict[str, str] = {}
        for pivot in self.pivot_langs:
            self.batched_prompts[pivot] = get_batched_back_translation_prompt(
                source_lang, pivot, num_variants
            )
        
        # Initialize OpenAI client (direct or Azure)
        self.use_openai_direct = os.getenv("USE_OPENAI_DIRECT", "false").lower() == "true"
        
        if self.use_openai_direct:
            # Use OpenAI directly
            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            logger.info(f"Using OpenAI direct API with model {self.model}")
        else:
            # Use Azure OpenAI
            self.client = AsyncAzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            )
            logger.info(f"Using Azure OpenAI with deployment {self.model}")
        
        # Statistics tracking
        self.statistics = AugmentationStatistics()
        
        logger.info(
            f"BackTranslator initialized: source={source_lang}, "
            f"pivots={self.pivot_langs}, model={self.model}, num_variants={num_variants}, "
            f"similarity_range=[{min_similarity}, {max_similarity}]"
        )
    
    async def _translate_single(
        self,
        text: str,
        system_prompt: str,
    ) -> str:
        """
        Translate a single text using the API.
        
        Args:
            text: Text to translate
            system_prompt: System prompt for translation direction
        
        Returns:
            Translated text
        """
        # Estimate tokens for rate limiting
        estimated_tokens = self.rate_limiter.estimate_request_tokens(
            text, self.max_tokens
        )
        
        async def make_request():
            # For O1/Reasoning models, the 'system' role is often not supported or has different behavior.
            # We merge system and user messages for these models.
            if "mini" in self.model.lower() or "o1" in self.model.lower():
                messages = [
                    {"role": "user", "content": f"{system_prompt}\n\nText to translate:\n{text}"},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,  # Use max_completion_tokens for newer models
            )
            content = response.choices[0].message.content.strip()
            usage = response.usage
            cached_tokens = 0
            if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
            
            if not content:
                logger.warning(f"Empty response received from model {self.model} for text: '{text[:50]}...'")
            else:
                logger.debug(f"Response from {self.model}: '{content[:50]}...'")
                
            return content, usage.prompt_tokens, usage.completion_tokens, cached_tokens
        
        return await self.rate_limiter.execute_with_retry(
            make_request,
            estimated_tokens=estimated_tokens,
        )
    
    async def _translate_batched_structured(
        self,
        text: str,
        pivot_lang: str,
    ) -> Tuple[str, List[str]]:
        """
        Perform batched back-translation with structured JSON output.
        
        Makes a single API call that:
        1. Translates source text to pivot language
        2. Generates N different back-translations
        3. Returns structured JSON with all results
        
        Args:
            text: Text to translate
            pivot_lang: Pivot language for back-translation
        
        Returns:
            Tuple of (pivot_translation, list of back_translations)
        """
        system_prompt = self.batched_prompts[pivot_lang]
        
        # Estimate tokens for rate limiting (more output expected for multiple variants)
        estimated_tokens = self.rate_limiter.estimate_request_tokens(
            text, self.max_tokens
        )
        
        async def make_request():
            # Combine system prompt and text for reasoning models
            if "mini" in self.model.lower() or "o1" in self.model.lower():
                messages = [
                    {"role": "user", "content": f"{system_prompt}\n\n{text}"},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            usage = response.usage
            cached_tokens = 0
            if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
            
            if not content:
                logger.warning(f"Empty response received from model {self.model} for text: '{text[:50]}...'")
                return None, usage.prompt_tokens, usage.completion_tokens, cached_tokens
            
            # Parse JSON response
            try:
                result = json.loads(content)
                pivot_translation = result.get("pivot_translation", "")
                back_translations = result.get("back_translations", [])
                
                if not isinstance(back_translations, list):
                    back_translations = [back_translations]
                
                logger.debug(
                    f"Batched BT: '{text[:30]}...' -> ({pivot_lang}) '{pivot_translation[:30]}...' "
                    f"-> {len(back_translations)} variants"
                )
                
                return (pivot_translation, back_translations), usage.prompt_tokens, usage.completion_tokens, cached_tokens
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}\nContent: {content[:200]}...")
                # Fallback: treat the entire response as a single back-translation
                return (content, [content]), usage.prompt_tokens, usage.completion_tokens, cached_tokens
        
        return await self.rate_limiter.execute_with_retry(
            make_request,
            estimated_tokens=estimated_tokens,
        )
    
    async def translate_batch(
        self,
        sentences: List[str],
        from_lang: str,
        to_lang: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """
        Translate a batch of sentences.
        
        Args:
            sentences: List of sentences to translate
            from_lang: Source language code
            to_lang: Target language code
            progress_callback: Optional callback(completed, total)
        
        Returns:
            List of translated sentences
        """
        # Get appropriate prompt
        if from_lang == self.source_lang:
            prompt = get_translation_prompts(from_lang, to_lang).system_forward
        else:
            prompt = get_translation_prompts(from_lang, to_lang).system_back
        
        results = []
        total = len(sentences)
        
        # Process in batches to control concurrency
        for i in range(0, total, self.batch_size):
            batch = sentences[i:i + self.batch_size]
            
            # Create tasks for concurrent execution
            tasks = [
                self._translate_single(text, prompt)
                for text in batch
            ]
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results, handling errors
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Translation failed for text {i+j}: {result}")
                    # Keep original text on failure
                    results.append(sentences[i + j])
                else:
                    results.append(result)
            
            if progress_callback:
                progress_callback(min(i + self.batch_size, total), total)
        
        return results
    
    async def back_translate_single(
        self,
        text: str,
        pivot_lang: str,
    ) -> Tuple[str, List[str]]:
        """
        Perform back-translation through a single pivot language.
        
        Uses batched structured output to get multiple variants in one API call.
        
        Args:
            text: Original text
            pivot_lang: Pivot language to use
        
        Returns:
            Tuple of (pivot_translation, list of back_translations)
        """
        result = await self._translate_batched_structured(text, pivot_lang)
        
        if result is None:
            logger.warning(f"Batched translation returned None for '{text[:50]}...'")
            return "", []
        
        pivot_text, back_texts = result
        return pivot_text, back_texts
    
    async def back_translate(
        self,
        sentences: List[str],
        progress_callback: Optional[Callable[[int, str], None]] = None,
        reset_statistics: bool = True,
    ) -> List[TranslationResult]:
        """
        Back-translate a list of sentences through all pivot languages in parallel.
        
        Uses batched structured output to get multiple variants per pivot in one API call.
        
        Args:
            sentences: List of sentences to augment
            progress_callback: Optional callback(increment, status)
            reset_statistics: Whether to reset statistics before starting
        
        Returns:
            List of TranslationResult objects with similarity scores
        """
        if reset_statistics:
            self.statistics = AugmentationStatistics()
        
        self.statistics.total_originals = len(sentences)
        
        total = len(sentences)
        
        async def process_item_pivot(
            text_idx: int, text: str, pivot: str
        ) -> Tuple[int, str, Optional[str], List[str], List[SimilarityScores], List[bool]]:
            """Process a single text through a single pivot, getting multiple variants."""
            try:
                pivot_text, back_texts = await self.back_translate_single(text, pivot)
                
                if not back_texts:
                    logger.warning(f"No back-translations returned for '{text[:50]}...' via {pivot}")
                    self.statistics.failed_translations += 1
                    if progress_callback:
                        progress_callback(1, f"{self.source_lang}→{pivot}→{self.source_lang} (NO VARIANTS)")
                    return text_idx, pivot, pivot_text, [], [], []
                
                # Compute similarity scores for each variant
                scores_list: List[SimilarityScores] = []
                is_dup_list: List[bool] = []
                
                for back_text in back_texts:
                    scores = compute_similarity(
                        text, back_text,
                        near_duplicate_threshold=self.max_similarity
                    )
                    scores_list.append(scores)
                    
                    # Check for duplicates and record statistics
                    if self.deduplicate:
                        is_kept = self.statistics.record_variant(
                            text, back_text, pivot, scores
                        )
                        is_dup = not is_kept
                        
                        # Also filter by minimum similarity
                        if scores.average < self.min_similarity:
                            is_dup = True
                    else:
                        is_dup = False
                        self.statistics.total_variants_generated += 1
                        self.statistics.variants_kept += 1
                        self.statistics.similarity_scores.append(scores.average)
                    
                    is_dup_list.append(is_dup)
                
                # Calculate average similarity for progress display
                avg_sim = sum(s.average for s in scores_list) / len(scores_list) if scores_list else 0
                kept_count = sum(1 for d in is_dup_list if not d)
                
                if progress_callback:
                    progress_callback(
                        1,
                        f"{self.source_lang}→{pivot}→{self.source_lang} ({kept_count}/{len(back_texts)} kept, sim={avg_sim:.2f})"
                    )
                
                return text_idx, pivot, pivot_text, back_texts, scores_list, is_dup_list
                
            except Exception as e:
                logger.error(f"Back-translation failed for '{text[:50]}...' via {pivot}: {e}")
                self.statistics.failed_translations += 1
                if progress_callback:
                    progress_callback(1, f"{self.source_lang}→{pivot}→{self.source_lang} (FAILED)")
                return text_idx, pivot, None, [], [], []

        # Create tasks for all sentence-pivot combinations
        tasks = []
        for i, text in enumerate(sentences):
            for pivot in self.pivot_langs:
                tasks.append(process_item_pivot(i, text, pivot))
        
        # Execute all combinations in parallel
        # Note: self.rate_limiter handles actual concurrency limits
        task_results = await asyncio.gather(*tasks)
        
        # Organize results back into TranslationResult objects
        # We use a dict to group by sentence index
        grouped_results = {i: {
            "pivot_translations": {},
            "back_translations": {},
            "similarity_scores": {},
            "is_duplicate": {}
        } for i in range(total)}
        
        for text_idx, pivot, p_text, b_texts, scores_list, is_dup_list in task_results:
            if p_text is not None:
                grouped_results[text_idx]["pivot_translations"][pivot] = p_text
                grouped_results[text_idx]["back_translations"][pivot] = b_texts
                grouped_results[text_idx]["similarity_scores"][pivot] = scores_list
                grouped_results[text_idx]["is_duplicate"][pivot] = is_dup_list
            else:
                # Handle failure case for this pivot
                grouped_results[text_idx]["back_translations"][pivot] = []
                grouped_results[text_idx]["similarity_scores"][pivot] = []
                grouped_results[text_idx]["is_duplicate"][pivot] = []
        
        # Construct final list of TranslationResults
        results = []
        for i, text in enumerate(sentences):
            results.append(TranslationResult(
                original=text,
                pivot_translations=grouped_results[i]["pivot_translations"],
                back_translations=grouped_results[i]["back_translations"],
                similarity_scores=grouped_results[i]["similarity_scores"],
                is_duplicate=grouped_results[i]["is_duplicate"],
            ))
            
        return results
    
    def get_statistics(self) -> AugmentationStatistics:
        """Get current augmentation statistics."""
        return self.statistics
    
    def reset_statistics(self) -> None:
        """Reset statistics for a new run."""
        self.statistics = AugmentationStatistics()
    
    async def augment_batch(
        self,
        sentences: List[str],
        include_original: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[List[str]]:
        """
        Augment a batch of sentences with back-translation variants.
        
        Args:
            sentences: List of sentences to augment
            include_original: Whether to include original in output
            progress_callback: Optional callback(completed, total, status)
        
        Returns:
            List of lists, each containing original + variants
        """
        results = await self.back_translate(sentences, progress_callback)
        
        augmented = []
        for result in results:
            if include_original:
                augmented.append(result.all_texts)
            else:
                augmented.append(result.variants)
        
        return augmented
    
    def filter_by_similarity(
        self,
        original: str,
        variants: List[str],
        min_similarity: Optional[float] = None,
        max_similarity: Optional[float] = None,
    ) -> Tuple[List[str], List[SimilarityScores]]:
        """
        Filter variants by similarity to original.
        
        Variants should be similar enough to preserve meaning but
        different enough to provide diversity.
        
        Args:
            original: Original text
            variants: List of variant texts
            min_similarity: Minimum similarity threshold (default: self.min_similarity)
            max_similarity: Maximum similarity threshold (default: self.max_similarity)
        
        Returns:
            Tuple of (filtered_variants, similarity_scores)
        """
        min_sim = min_similarity if min_similarity is not None else self.min_similarity
        max_sim = max_similarity if max_similarity is not None else self.max_similarity
        
        filtered = []
        scores = []
        
        for variant in variants:
            sim = compute_similarity(original, variant, near_duplicate_threshold=max_sim)
            
            # Skip exact duplicates
            if sim.is_exact_duplicate:
                logger.debug(f"Filtered exact duplicate: '{variant[:30]}...'")
                continue
            
            # Skip near duplicates
            if sim.is_near_duplicate:
                logger.debug(f"Filtered near duplicate (sim={sim.average:.2f}): '{variant[:30]}...'")
                continue
            
            # Check similarity range
            if sim.average < min_sim:
                logger.debug(
                    f"Filtered variant (sim={sim.average:.2f} < {min_sim}): "
                    f"'{variant[:30]}...' for '{original[:30]}...'"
                )
                continue
            
            if sim.average > max_sim:
                logger.debug(
                    f"Filtered variant (sim={sim.average:.2f} > {max_sim}): "
                    f"'{variant[:30]}...' for '{original[:30]}...'"
                )
                continue
            
            filtered.append(variant)
            scores.append(sim)
        
        return filtered, scores


def create_back_translator_for_dataset(
    dataset_id: str,
    rate_limiter: Optional[AzureRateLimiter] = None,
    **kwargs,
) -> BackTranslator:
    """
    Create a BackTranslator configured for a specific dataset.
    
    Args:
        dataset_id: Dataset identifier (e.g., 'RWTH_PHOENIX_2014T')
        rate_limiter: Optional rate limiter instance
        **kwargs: Additional arguments for BackTranslator
    
    Returns:
        Configured BackTranslator instance
    """
    source_lang = get_source_language(dataset_id)
    
    return BackTranslator(
        source_lang=source_lang,
        rate_limiter=rate_limiter,
        **kwargs,
    )


async def back_translate_dataframe(
    df,
    text_column: str,
    translator: BackTranslator,
    output_column: str = "augmented_texts",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
):
    """
    Back-translate texts in a pandas DataFrame.
    
    Args:
        df: Input DataFrame
        text_column: Name of column containing texts
        translator: BackTranslator instance
        output_column: Name of column for augmented texts
        progress_callback: Optional progress callback
    
    Returns:
        DataFrame with augmented texts column
    """
    import pandas as pd
    
    texts = df[text_column].tolist()
    augmented = await translator.augment_batch(
        texts,
        include_original=True,
        progress_callback=progress_callback,
    )
    
    df = df.copy()
    df[output_column] = augmented
    
    return df


async def expand_dataframe_with_variants(
    df,
    text_column: str,
    translator: BackTranslator,
    id_column: str = "id",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
):
    """
    Expand DataFrame by creating new rows for each augmented variant.
    
    Args:
        df: Input DataFrame
        text_column: Name of column containing texts
        translator: BackTranslator instance
        id_column: Column to use for tracking originals
        progress_callback: Optional progress callback
    
    Returns:
        Expanded DataFrame with augmented rows
    """
    import pandas as pd
    
    texts = df[text_column].tolist()
    results = await translator.back_translate(texts, progress_callback)
    
    new_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        result = results[i]
        
        # Original row
        new_rows.append(row.to_dict())
        
        # Variant rows
        for pivot, variant in result.back_translations.items():
            variant_row = row.to_dict()
            variant_row[text_column] = variant
            variant_row["augmentation_source"] = f"back_translate_{pivot}"
            new_rows.append(variant_row)
    
    return pd.DataFrame(new_rows)
