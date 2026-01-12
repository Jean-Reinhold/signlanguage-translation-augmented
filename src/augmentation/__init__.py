"""
Text Augmentation Module for Sign Language Translation.

This module provides tools for augmenting SLT datasets using:
- Back-translation through pivot languages
- Paraphrasing with LLMs
- Rate-limited Azure OpenAI API access

Main components:
- BackTranslator: Core back-translation engine
- AzureRateLimiter: Rate limiting with token bucket algorithm
- prompts: Language-specific translation prompts
- augment_dataset: CLI for batch augmentation

Example usage:
    from src.augmentation import BackTranslator, AzureRateLimiter
    
    rate_limiter = AzureRateLimiter(rpm=60, tpm=90000)
    translator = BackTranslator(
        source_lang="de",
        rate_limiter=rate_limiter
    )
    
    results = await translator.back_translate(["Guten Morgen"])
"""

from .back_translate import (
    BackTranslator,
    TranslationResult,
    SimilarityScores,
    AugmentationStatistics,
    create_back_translator_for_dataset,
    back_translate_dataframe,
    expand_dataframe_with_variants,
    compute_similarity,
    jaccard_similarity,
    word_overlap_similarity,
    levenshtein_similarity,
    ngram_similarity,
)
from .prompts import (
    DATASET_LANGUAGES,
    PIVOT_LANGUAGES,
    get_dataset_prompts,
    get_source_language,
    get_pivot_languages,
    get_paraphrase_prompt,
    get_translation_prompts,
)
from .rate_limiter import (
    AzureRateLimiter,
    RateLimiterMetrics,
    create_rate_limiter_from_env,
)

__all__ = [
    # Back-translation
    "BackTranslator",
    "TranslationResult",
    "SimilarityScores",
    "AugmentationStatistics",
    "create_back_translator_for_dataset",
    "back_translate_dataframe",
    "expand_dataframe_with_variants",
    # Similarity functions
    "compute_similarity",
    "jaccard_similarity",
    "word_overlap_similarity",
    "levenshtein_similarity",
    "ngram_similarity",
    # Prompts
    "DATASET_LANGUAGES",
    "PIVOT_LANGUAGES",
    "get_dataset_prompts",
    "get_source_language",
    "get_pivot_languages",
    "get_paraphrase_prompt",
    "get_translation_prompts",
    # Rate limiting
    "AzureRateLimiter",
    "RateLimiterMetrics",
    "create_rate_limiter_from_env",
]
