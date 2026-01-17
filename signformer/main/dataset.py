# coding: utf-8
"""
Data module
"""
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple, Iterator, Optional
import logging
import pickle
import gzip
import torch
import os


def load_dataset_file(filename):
    """Load a dataset gzip file that may contain multiple pickled chunks.

    Supports both single-pickle files (original format) and multi-pickle
    streams produced by concatenating chunks (multipart merged files).
    Also supports .part* files where a dataset is split into multiple parts.
    """
    import glob
    import os
    
    # Check if file exists directly or as parts
    if os.path.exists(filename):
        files_to_read = [filename]
    else:
        # Look for .part* files
        part_pattern = filename + ".part*"
        files_to_read = sorted(glob.glob(part_pattern), 
                               key=lambda x: int(x.rsplit('.part', 1)[1]) if '.part' in x else 0)
        if not files_to_read:
            raise FileNotFoundError(f"No file or parts found for: {filename}")
    
    samples = []
    for part_file in files_to_read:
        with gzip.open(part_file, "rb") as f:
            while True:
                try:
                    obj = pickle.load(f)
                except EOFError:
                    break
                # Each chunk can be a list of samples or a single sample
                if isinstance(obj, list):
                    samples.extend(obj)
                else:
                    samples.append(obj)
    return samples


def iter_dataset_file(filename) -> Iterator[dict]:
    """Iterate over a dataset gzip file yielding samples one by one.

    Works with both single-pickle files (list of samples) and multi-pickle
    streams (several lists written sequentially). Also supports .part* files
    where a dataset is split into multiple part files.
    This avoids loading the entire dataset into RAM at once.
    """
    import glob
    import os
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Check if file exists directly or as parts
    if os.path.exists(filename):
        files_to_read = [filename]
    else:
        # Look for .part* files
        part_pattern = filename + ".part*"
        files_to_read = sorted(glob.glob(part_pattern), 
                               key=lambda x: int(x.rsplit('.part', 1)[1]) if '.part' in x else 0)
        if not files_to_read:
            raise FileNotFoundError(f"No file or parts found for: {filename}")
        logger.info("[dataset] Found %d parts for %s", len(files_to_read), filename)
    
    total = 0
    chunks = 0
    for part_file in files_to_read:
        logger.info("[dataset] Open %s", part_file)
        with gzip.open(part_file, "rb") as f:
            while True:
                try:
                    obj = pickle.load(f)
                except EOFError:
                    break
                chunks += 1
                if isinstance(obj, list):
                    for s in obj:
                        total += 1
                        if total % 50000 == 0:
                            logger.info("[dataset] %s: yielded %d samples (chunks=%d)", filename, total, chunks)
                        yield s
                else:
                    total += 1
                    if total % 50000 == 0:
                        logger.info("[dataset] %s: yielded %d samples (chunks=%d)", filename, total, chunks)
                    yield obj
    logger.info("[dataset] Done %s: total samples %d, chunks %d", filename, total, chunks)


def dataset_name_from_path(path: str) -> str:
    """Extract dataset name from a dataset file path."""
    base = os.path.basename(path)
    if ".pami0" in base:
        base = base.split(".pami0")[0]
    else:
        base = base.split(".")[0]
    return base


def _normalize_dataset_name(name: str) -> str:
    for suffix in ("-aug-1024", "-1024-aug", "-aug", "-1024"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def lookup_lang_token(dataset_name: str, lang_token_map: dict) -> Optional[str]:
    """Lookup a language token for a dataset name with simple normalization."""
    if not lang_token_map:
        return None
    candidates = [dataset_name, _normalize_dataset_name(dataset_name)]
    for c in list(candidates):
        candidates.extend([c.lower(), c.upper()])
    for key in candidates:
        if key in lang_token_map:
            return lang_token_map[key]
    return None


def apply_lang_token(text: str, lang_token: Optional[str]) -> str:
    """Prefix language token to text if provided and not already present."""
    if not lang_token:
        return text
    token = lang_token.strip()
    if not token:
        return text
    if text.startswith(token):
        return text
    return f"{token} {text}".strip()


def normalize_sign_features(sign: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad or crop sign feature dimension to target_dim."""
    if sign is None or target_dim is None:
        return sign
    current_dim = sign.shape[-1]
    if current_dim == target_dim:
        return sign
    if current_dim > target_dim:
        return sign[..., :target_dim]
    pad_shape = list(sign.shape)
    pad_shape[-1] = target_dim - current_dim
    pad = torch.zeros(*pad_shape, dtype=sign.dtype, device=sign.device)
    return torch.cat([sign, pad], dim=-1)


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        txt_prefix: Optional[str] = None,
        sign_feature_size: Optional[int] = None,
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            txt_prefix: Optional token to prefix target text (e.g., language tag).
            sign_feature_size: Optional target feature dimension for sign tensors.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]

        if not isinstance(path, list):
            path = [path]

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.info("[dataset] Building SignTranslationDataset from %d file(s)", len(path))
        samples = {}
        processed = 0
        for annotation_file in path:
            logger.info("[dataset] Reading %s", annotation_file)
            for s in iter_dataset_file(annotation_file):
                seq_id = s["name"]
                text = s["text"].strip()
                text = apply_lang_token(text, txt_prefix)
                sign = normalize_sign_features(s["sign"], sign_feature_size)
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == text
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], sign], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": text,
                        "sign": sign,
                    }
                processed += 1
                if processed % 100000 == 0:
                    logger.info("[dataset] Aggregated %d samples so far", processed)

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        logger.info("[dataset] Built %d examples", len(examples))
        super().__init__(examples, fields, **kwargs)

class SignTranslationDataset3D(data.Dataset):
    """Defines a dataset for machine translation based on 3D keypoint data."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, Field, Field, Field],
        **kwargs
    ):

        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("name", fields[0]),   # Name of the sample
                ("sgn", fields[1]),    # Keypoint data
                ("gls", fields[2]),    # Gloss
                ("txt", fields[3]),    # Text
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path:
            for s in iter_dataset_file(annotation_file):
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    # Concatenate keypoint data along the num_frames axis (0)
                    samples[seq_id]["keypoint"] = torch.cat(
                        [samples[seq_id]["keypoint"], s["keypoint"]], axis=0
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "keypoint": s["keypoint"],  # Store 3D keypoint data
                    }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        # Adding small value for numerical stability
                        sample["keypoint"] + 1e-8,  # Keypoint data (num_frames, 133, 3)
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
