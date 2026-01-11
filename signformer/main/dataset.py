# coding: utf-8
"""
Data module
"""
from torchtext.legacy import data
from torchtext.legacy.data import Field, RawField
from typing import List, Tuple, Iterator
import logging
import pickle
import gzip
import torch


def load_dataset_file(filename):
    """Load a dataset gzip file that may contain multiple pickled chunks.

    Supports both single-pickle files (original format) and multi-pickle
    streams produced by concatenating chunks (multipart merged files).
    """
    samples = []
    with gzip.open(filename, "rb") as f:
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
    streams (several lists written sequentially). This avoids loading the
    entire dataset into RAM at once.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info("[dataset] Open %s", filename)
    total = 0
    chunks = 0
    with gzip.open(filename, "rb") as f:
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


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
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
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
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
