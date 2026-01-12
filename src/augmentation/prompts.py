"""
Language-specific prompts for back-translation augmentation.

This module provides carefully crafted prompts for:
1. Forward translation (source → pivot language)
2. Back-translation (pivot → source language)

Following best practices:
- Preserve meaning, tense, and register
- Keep length within ±3 tokens of original
- No information addition or omission
- Register and formality preservation
"""

from dataclasses import dataclass
from typing import Dict, Tuple

# ISO 639-1 language codes
LANG_CODES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "el": "Greek",
}

# Dataset to language mapping
DATASET_LANGUAGES: Dict[str, str] = {
    "RWTH_PHOENIX_2014T": "de",
    "phoenix14t": "de",
    "lsat": "es",
    "LSA-T": "es",
    "How2Sign": "en",
    "ISL": "en",
    "indian_sl_dataset": "en",
    "LSFB-CONT": "fr",
    "GSL": "el",
    "greek_sl_dataset": "el",
}

# Pivot language policy:
# - Use English as pivot for all languages
# - Use Spanish as pivot when source is English (can't pivot through itself)
PIVOT_LANGUAGES: Dict[str, Tuple[str, ...]] = {
    "de": ("en",),           # German → English pivot
    "en": ("es",),           # English → Spanish pivot
    "es": ("en",),           # Spanish → English pivot
    "fr": ("en",),           # French → English pivot
    "el": ("en",),           # Greek → English pivot
}


@dataclass
class TranslationPrompt:
    """Container for forward and back-translation prompts."""
    system_forward: str
    system_back: str
    source_lang: str
    pivot_lang: str


def get_forward_translation_prompt(source_lang: str, target_lang: str) -> str:
    """
    Generate system prompt for forward translation (source → pivot).
    
    The forward translation should be literal and preserve all nuances.
    """
    source_name = LANG_CODES.get(source_lang, source_lang)
    target_name = LANG_CODES.get(target_lang, target_lang)
    
    return f"""You are a professional translator specializing in {source_name} to {target_name} translation.

Your task is to translate the user's {source_name} text into {target_name}.

CRITICAL RULES:
1. Produce a LITERAL, ACCURATE translation that preserves ALL meaning
2. Maintain the EXACT same tense (past, present, future)
3. Preserve the register (formal/informal) and tone
4. Keep the translation length similar to the original (within ±3 words)
5. Do NOT add any information not present in the original
6. Do NOT omit any information from the original
7. Do NOT add explanations, notes, or alternatives
8. Return ONLY the translated text, nothing else

Translate the following {source_name} text to {target_name}:"""


def get_back_translation_prompt(source_lang: str, target_lang: str) -> str:
    """
    Generate system prompt for back-translation (pivot → source).
    
    The back-translation can introduce natural variation while preserving meaning.
    """
    source_name = LANG_CODES.get(source_lang, source_lang)
    target_name = LANG_CODES.get(target_lang, target_lang)
    
    return f"""You are a professional translator specializing in {source_name} to {target_name} translation.

Your task is to translate the user's {source_name} text into {target_name}, creating a NATURAL rephrasing.

CRITICAL RULES:
1. Produce a NATURAL, FLUENT translation in {target_name}
2. The meaning must be EXACTLY preserved - no additions or omissions
3. Maintain the EXACT same tense (past, present, future)
4. Preserve the register (formal/informal) and tone
5. You MAY use synonyms and rephrase for natural flow
6. Keep the translation length similar to the original (within ±3 words)
7. Do NOT add explanations, notes, or alternatives
8. Return ONLY the translated text, nothing else

Translate the following {source_name} text to {target_name}:"""


# Language-specific translation prompts with cultural/linguistic considerations
LANGUAGE_SPECIFIC_NOTES: Dict[str, str] = {
    "de": """
Additional German notes:
- Preserve compound word structure where appropriate
- Maintain correct case (nominative, accusative, dative, genitive)
- Keep formal "Sie" vs informal "du" distinction""",
    
    "es": """
Additional Spanish notes:
- Preserve subjunctive mood where used
- Maintain correct gender agreement
- Keep formal "usted" vs informal "tú" distinction""",
    
    "en": """
Additional English notes:
- Preserve British vs American spelling if evident
- Maintain contractions or full forms as in original
- Keep formal vs casual register""",
    
    "fr": """
Additional French notes:
- Preserve formal "vous" vs informal "tu" distinction
- Maintain correct gender agreement
- Keep liaison patterns where applicable""",
    
    "el": """
Additional Greek notes:
- Preserve formal vs informal address
- Maintain correct case endings
- Keep accent marks in proper positions""",
}


def get_translation_prompts(
    source_lang: str,
    pivot_lang: str,
    include_language_notes: bool = True
) -> TranslationPrompt:
    """
    Get complete translation prompts for a language pair.
    
    Args:
        source_lang: Source language code (e.g., 'de', 'en')
        pivot_lang: Pivot language code (e.g., 'es', 'en')
        include_language_notes: Whether to include language-specific notes
    
    Returns:
        TranslationPrompt with forward and back-translation system prompts
    """
    forward_prompt = get_forward_translation_prompt(source_lang, pivot_lang)
    back_prompt = get_back_translation_prompt(pivot_lang, source_lang)
    
    if include_language_notes:
        if source_lang in LANGUAGE_SPECIFIC_NOTES:
            forward_prompt += LANGUAGE_SPECIFIC_NOTES[source_lang]
        if pivot_lang in LANGUAGE_SPECIFIC_NOTES:
            back_prompt += LANGUAGE_SPECIFIC_NOTES[source_lang]
    
    return TranslationPrompt(
        system_forward=forward_prompt,
        system_back=back_prompt,
        source_lang=source_lang,
        pivot_lang=pivot_lang,
    )


def get_dataset_prompts(dataset_id: str) -> Dict[str, TranslationPrompt]:
    """
    Get all translation prompts needed for a dataset.
    
    Args:
        dataset_id: Dataset identifier (e.g., 'RWTH_PHOENIX_2014T')
    
    Returns:
        Dict mapping pivot language to TranslationPrompt
    """
    # Normalize dataset ID
    normalized_id = dataset_id
    for key in DATASET_LANGUAGES:
        if key.lower() == dataset_id.lower():
            normalized_id = key
            break
    
    source_lang = DATASET_LANGUAGES.get(normalized_id)
    if source_lang is None:
        raise ValueError(
            f"Unknown dataset: {dataset_id}. "
            f"Known datasets: {list(DATASET_LANGUAGES.keys())}"
        )
    
    pivot_langs = PIVOT_LANGUAGES.get(source_lang, ("en",))
    
    prompts = {}
    for pivot in pivot_langs:
        prompts[pivot] = get_translation_prompts(source_lang, pivot)
    
    return prompts


def get_paraphrase_prompt(language: str, n_variants: int = 2) -> str:
    """
    Get system prompt for direct paraphrasing (no pivot language).
    
    This is an alternative to back-translation for generating variants.
    
    Args:
        language: Language code for paraphrasing
        n_variants: Number of paraphrases to generate
    
    Returns:
        System prompt string
    """
    lang_name = LANG_CODES.get(language, language)
    
    return f"""You are a helpful assistant rewriting {lang_name} sentences.

For every user input, produce {n_variants} paraphrases that:
• Preserve the EXACT meaning, tense and register
• Reuse at least 70% of the original words (higher is better)
• Vary mainly through word order or minor synonym substitutions
• Keep length within ±3 tokens of the original
• Do NOT add or omit information
• Do NOT add numbering or explanations

Return only the paraphrases, one per line."""


def get_batched_back_translation_prompt(
    source_lang: str,
    pivot_lang: str,
    n_variants: int = 2,
    include_language_notes: bool = True
) -> str:
    """
    Get prompt for batched back-translation that returns N variants in JSON format.
    
    This performs forward translation (source → pivot) and multiple back-translations
    (pivot → source) in a single API call, returning structured JSON output.
    
    Args:
        source_lang: Source language code (e.g., 'de', 'en')
        pivot_lang: Pivot language code (e.g., 'es', 'en')
        n_variants: Number of back-translation variants to generate
        include_language_notes: Whether to include language-specific notes
    
    Returns:
        System prompt string for JSON structured output
    """
    source_name = LANG_CODES.get(source_lang, source_lang)
    pivot_name = LANG_CODES.get(pivot_lang, pivot_lang)
    
    language_notes = ""
    if include_language_notes and source_lang in LANGUAGE_SPECIFIC_NOTES:
        language_notes = LANGUAGE_SPECIFIC_NOTES[source_lang]
    
    return f"""You are a professional translator specializing in {source_name} and {pivot_name}.

Your task is to perform back-translation augmentation:
1. First, translate the input {source_name} text to {pivot_name}
2. Then, create {n_variants} DIFFERENT back-translations from {pivot_name} to {source_name}

CRITICAL RULES:
• Each back-translation must preserve the EXACT meaning of the original
• Maintain the same tense (past, present, future) in all variants
• Preserve the register (formal/informal) and tone
• Each variant should use different word choices or sentence structure
• Keep each variant's length within ±3 words of the original
• Do NOT add or omit any information
{language_notes}

You MUST respond with valid JSON in this exact format:
{{
  "pivot_translation": "<the {pivot_name} translation>",
  "back_translations": ["<variant 1 in {source_name}>", "<variant 2 in {source_name}>", ...]
}}

Translate the following {source_name} text:"""


# Convenience function for getting source language from dataset
def get_source_language(dataset_id: str) -> str:
    """Get the source language code for a dataset."""
    for key, lang in DATASET_LANGUAGES.items():
        if key.lower() == dataset_id.lower():
            return lang
    raise ValueError(f"Unknown dataset: {dataset_id}")


def get_pivot_languages(source_lang: str) -> Tuple[str, ...]:
    """Get pivot languages for a source language."""
    return PIVOT_LANGUAGES.get(source_lang, ("en",))
