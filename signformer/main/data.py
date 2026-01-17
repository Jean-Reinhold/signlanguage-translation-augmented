# coding: utf-8
"""
Data module
"""
import os
import sys
import random

import torch
from torchtext import data
from torchtext.data import Dataset, Iterator
import socket
from main.dataset import (
    SignTranslationDataset,
    iter_dataset_file,
    dataset_name_from_path,
    lookup_lang_token,
    apply_lang_token,
)
import logging
from main.vocabulary import (
    build_vocab,
    Vocabulary,
    UNK_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
    PAD_TOKEN,
)


def load_data(data_cfg: dict) -> (Dataset, Dataset, Dataset, Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    If you set ``random_dev_subset``, a random selection of this size is used
    from the dev development instead of the full development set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuration file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: test dataset if given, otherwise None
        - gls_vocab: gloss vocabulary extracted from training data
        - txt_vocab: spoken text vocabulary extracted from training data
    """

    data_path = data_cfg.get("data_path", "./data")

    if isinstance(data_cfg["train"], list):
        train_paths = [os.path.join(data_path, x) for x in data_cfg["train"]]
        dev_paths = [os.path.join(data_path, x) for x in data_cfg["dev"]]
        test_paths = [os.path.join(data_path, x) for x in data_cfg["test"]]
        # Handle both list (multimodal) and int (multi-dataset same features) for feature_size
        if isinstance(data_cfg["feature_size"], list):
            pad_feature_size = sum(data_cfg["feature_size"])
        else:
            pad_feature_size = data_cfg["feature_size"]

    else:
        train_paths = os.path.join(data_path, data_cfg["train"])
        dev_paths = os.path.join(data_path, data_cfg["dev"])
        test_paths = os.path.join(data_path, data_cfg["test"])
        pad_feature_size = data_cfg["feature_size"]

    level = data_cfg["level"]
    txt_lowercase = data_cfg["txt_lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    def tokenize_text(text):
        if level == "char":
            return list(text)
        else:
            return text.split()

    def tokenize_features(features):
        ft_list = torch.split(features, 1, dim=0)
        return [ft.squeeze() for ft in ft_list]

    # NOTE (Cihan): The something was necessary to match the function signature.

    def stack_features(features, something):
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)

    sequence_field = data.RawField()
    signer_field = data.RawField()

    sgn_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        preprocessing=tokenize_features,
        tokenize=lambda features: features,  # TODO (Cihan): is this necessary?
        batch_first=True,
        include_lengths=True,
        postprocessing=stack_features,
        pad_token=torch.zeros((pad_feature_size,)),
    )

    gls_field = data.Field(
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        batch_first=True,
        lower=False,
        include_lengths=True,
    )

    txt_field = data.Field(
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        unk_token=UNK_TOKEN,
        batch_first=True,
        lower=txt_lowercase,
        include_lengths=True,
    )

    stream_train_parts = data_cfg.get("stream_train_parts", False)
    prepend_lang_token = data_cfg.get("prepend_lang_token", False)
    lang_token_map = data_cfg.get("lang_token_map", {}) or {}
    missing_lang_token = set()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # If streaming, build vocabs without materializing the full train dataset
    # to avoid excessive RAM usage.
    if stream_train_parts:
        logger.info("[data] stream_train_parts enabled")
        max_vocab_samples = None
        gls_max_size = data_cfg.get("gls_voc_limit", sys.maxsize)
        gls_min_freq = data_cfg.get("gls_voc_min_freq", 1)
        txt_max_size = data_cfg.get("txt_voc_limit", sys.maxsize)
        txt_min_freq = data_cfg.get("txt_voc_min_freq", 1)

        gls_vocab_file = data_cfg.get("gls_vocab", None)
        txt_vocab_file = data_cfg.get("txt_vocab", None)

        # Tokenize helper for streaming
        def _yield_tokens(paths, field_name):
            import glob as _glob
            yielded = 0
            samples_seen = 0
            for p in paths if isinstance(paths, list) else [paths]:
                dataset_name = dataset_name_from_path(p)
                lang_token = None
                if prepend_lang_token:
                    lang_token = lookup_lang_token(dataset_name, lang_token_map)
                    if lang_token is None and dataset_name not in missing_lang_token:
                        logger.warning(
                            "[data] No lang token mapping for dataset '%s'", dataset_name
                        )
                        missing_lang_token.add(dataset_name)
                # Check if file exists directly or as parts
                import gzip as _gzip
                import pickle as _pickle
                if os.path.exists(p):
                    files_to_read = [p]
                else:
                    # Look for .part* files
                    part_pattern = p + ".part*"
                    files_to_read = sorted(_glob.glob(part_pattern), key=lambda x: int(x.rsplit('.part', 1)[1]) if '.part' in x else 0)
                    if not files_to_read:
                        raise FileNotFoundError(f"No file or parts found for: {p}")
                    logger.info("[data] Found %d parts for %s", len(files_to_read), p)
                
                for part_file in files_to_read:
                    logger.info("[data] Streaming tokens from %s (%s)", part_file, field_name)
                    # Read gzip pickles chunk-by-chunk and drop heavy fields immediately
                    with _gzip.open(part_file, "rb") as _f:
                        while True:
                            try:
                                _obj = _pickle.load(_f)
                            except EOFError:
                                break
                            # _obj can be a list of samples or a single sample
                            if isinstance(_obj, list):
                                iterable = _obj
                            else:
                                iterable = [_obj]
                            for _s in iterable:
                                samples_seen += 1
                                # extract text/gloss, then drop heavy tensor to free RAM early
                                _text = _s["text"].strip() if field_name == "txt" else _s["gloss"].strip()
                                if field_name == "txt" and lang_token:
                                    _text = apply_lang_token(_text, lang_token)
                                if "sign" in _s:
                                    try:
                                        del _s["sign"]
                                    except Exception:
                                        pass
                                if level == "char":
                                    _tokens = list(_text)
                                else:
                                    _tokens = _text.split()
                                for t in _tokens:
                                    yielded += 1
                                    if yielded % 1000000 == 0:
                                        logger.info("[data] Streamed %d tokens (%s)", yielded, field_name)
                                    yield t

        # Build vocabs either from files or by streaming tokens
        if gls_vocab_file is not None:
            gls_vocab = build_vocab(
                field="gls",
                min_freq=gls_min_freq,
                max_size=gls_max_size,
                dataset=None,
                vocab_file=gls_vocab_file,
            )
        else:
            # Low-memory heavy-hitters (Misra-Gries) to get candidate set of size <= gls_max_size
            def _heavy_hitters_candidates(k):
                counters = {}
                for tok in _yield_tokens(train_paths, "gls"):
                    if tok in counters:
                        counters[tok] += 1
                    elif len(counters) < max(1, k - 1):
                        counters[tok] = 1
                    else:
                        # decrement all
                        to_del = []
                        for t in counters:
                            counters[t] -= 1
                            if counters[t] == 0:
                                to_del.append(t)
                        for t in to_del:
                            del counters[t]
                return set(counters.keys())

            def _count_exact_for(tokens_set):
                from collections import Counter

                exact = Counter()
                if not tokens_set:
                    return exact
                for tok in _yield_tokens(train_paths, "gls"):
                    if tok in tokens_set:
                        exact[tok] += 1
                return exact

            candidates = _heavy_hitters_candidates(gls_max_size)
            gls_counter = _count_exact_for(candidates)
            if gls_min_freq > -1:
                gls_counter = type(gls_counter)({t: c for t, c in gls_counter.items() if c >= gls_min_freq})
            tokens_and_frequencies = sorted(gls_counter.items(), key=lambda tup: tup[0])
            tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
            gls_tokens = [i[0] for i in tokens_and_frequencies[:gls_max_size]]
            from main.vocabulary import GlossVocabulary

            gls_vocab = GlossVocabulary(tokens=gls_tokens)

        if txt_vocab_file is not None:
            txt_vocab = build_vocab(
                field="txt",
                min_freq=txt_min_freq,
                max_size=txt_max_size,
                dataset=None,
                vocab_file=txt_vocab_file,
            )
        else:
            # Low-memory heavy-hitters for text vocabulary
            def _heavy_hitters_candidates_txt(k):
                counters = {}
                for tok in _yield_tokens(train_paths, "txt"):
                    if tok in counters:
                        counters[tok] += 1
                    elif len(counters) < max(1, k - 1):
                        counters[tok] = 1
                    else:
                        to_del = []
                        for t in counters:
                            counters[t] -= 1
                            if counters[t] == 0:
                                to_del.append(t)
                        for t in to_del:
                            del counters[t]
                return set(counters.keys())

            def _count_exact_for_txt(tokens_set):
                from collections import Counter

                exact = Counter()
                if not tokens_set:
                    return exact
                for tok in _yield_tokens(train_paths, "txt"):
                    if tok in tokens_set:
                        exact[tok] += 1
                return exact

            candidates = _heavy_hitters_candidates_txt(txt_max_size)
            txt_counter = _count_exact_for_txt(candidates)
            if txt_min_freq > -1:
                txt_counter = type(txt_counter)({t: c for t, c in txt_counter.items() if c >= txt_min_freq})
            tokens_and_frequencies = sorted(txt_counter.items(), key=lambda tup: tup[0])
            tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
            txt_tokens = [i[0] for i in tokens_and_frequencies[:txt_max_size]]
            from main.vocabulary import TextVocabulary

            txt_vocab = TextVocabulary(tokens=txt_tokens)

        # assign vocabs to fields
        gls_field.vocab = gls_vocab
        txt_field.vocab = txt_vocab

        # Don't instantiate full train dataset; return a descriptor for streaming
        train_data = {
            "paths": train_paths,
            "fields": (sequence_field, signer_field, sgn_field, gls_field, txt_field),
            "prepend_lang_token": prepend_lang_token,
            "lang_token_map": lang_token_map,
            "stream_chunk_size": data_cfg.get("stream_chunk_size", 5000),
        }
        logger.info("[data] Returning streaming train descriptor (no full Dataset loaded)")
    else:
        txt_prefix = None
        if prepend_lang_token and isinstance(train_paths, str):
            dataset_name = dataset_name_from_path(train_paths)
            txt_prefix = lookup_lang_token(dataset_name, lang_token_map)
            if txt_prefix is None:
                logger.warning(
                    "[data] No lang token mapping for dataset '%s'", dataset_name
                )
        train_data = SignTranslationDataset(
            path=train_paths,
            fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
            txt_prefix=txt_prefix,
            sign_feature_size=pad_feature_size,
            filter_pred=lambda x: len(vars(x)["sgn"]) <= max_sent_length
            and len(vars(x)["txt"]) <= max_sent_length,
        )

    gls_max_size = data_cfg.get("gls_voc_limit", sys.maxsize)
    gls_min_freq = data_cfg.get("gls_voc_min_freq", 1)
    txt_max_size = data_cfg.get("txt_voc_limit", sys.maxsize)
    txt_min_freq = data_cfg.get("txt_voc_min_freq", 1)

    gls_vocab_file = data_cfg.get("gls_vocab", None)
    txt_vocab_file = data_cfg.get("txt_vocab", None)

    if not stream_train_parts:
        gls_vocab = build_vocab(
            field="gls",
            min_freq=gls_min_freq,
            max_size=gls_max_size,
            dataset=train_data,
            vocab_file=gls_vocab_file,
        )
        txt_vocab = build_vocab(
            field="txt",
            min_freq=txt_min_freq,
            max_size=txt_max_size,
            dataset=train_data,
            vocab_file=txt_vocab_file,
        )
    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        # select this many training examples randomly and discard the rest
        keep_ratio = random_train_subset / len(train_data)
        keep, _ = train_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
        )
        train_data = keep

    stream_dev_parts = data_cfg.get("stream_dev_parts", stream_train_parts)
    if stream_dev_parts:
        dev_data = {
            "paths": dev_paths,
            "fields": (sequence_field, signer_field, sgn_field, gls_field, txt_field),
            "prepend_lang_token": prepend_lang_token,
            "lang_token_map": lang_token_map,
            "stream_chunk_size": data_cfg.get("stream_chunk_size", 5000),
        }
    else:
        txt_prefix = None
        if prepend_lang_token and isinstance(dev_paths, str):
            dataset_name = dataset_name_from_path(dev_paths)
            txt_prefix = lookup_lang_token(dataset_name, lang_token_map)
            if txt_prefix is None:
                logger.warning(
                    "[data] No lang token mapping for dataset '%s'", dataset_name
                )
        dev_data = SignTranslationDataset(
            path=dev_paths,
            fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
            txt_prefix=txt_prefix,
            sign_feature_size=pad_feature_size,
        )
    random_dev_subset = data_cfg.get("random_dev_subset", -1)
    if random_dev_subset > -1:
        # select this many development examples randomly and discard the rest
        keep_ratio = random_dev_subset / len(dev_data)
        keep, _ = dev_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
        )
        dev_data = keep

    # check if target exists
    stream_test_parts = data_cfg.get("stream_test_parts", stream_train_parts)
    if stream_test_parts:
        test_data = {
            "paths": test_paths,
            "fields": (sequence_field, signer_field, sgn_field, gls_field, txt_field),
            "prepend_lang_token": prepend_lang_token,
            "lang_token_map": lang_token_map,
            "stream_chunk_size": data_cfg.get("stream_chunk_size", 5000),
        }
    else:
        txt_prefix = None
        if prepend_lang_token and isinstance(test_paths, str):
            dataset_name = dataset_name_from_path(test_paths)
            txt_prefix = lookup_lang_token(dataset_name, lang_token_map)
            if txt_prefix is None:
                logger.warning(
                    "[data] No lang token mapping for dataset '%s'", dataset_name
                )
        test_data = SignTranslationDataset(
            path=test_paths,
            fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
            txt_prefix=txt_prefix,
            sign_feature_size=pad_feature_size,
        )

    if not stream_train_parts:
        gls_field.vocab = gls_vocab
        txt_field.vocab = txt_vocab
    return train_data, dev_data, test_data, gls_vocab, txt_vocab


# TODO (Cihan): I don't like this use of globals.
#  Need to find a more elegant solution for this it at some point.
# pylint: disable=global-at-module-level
global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)"""
    global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch
    if count == 1:
        max_sgn_in_batch = 0
        max_gls_in_batch = 0
        max_txt_in_batch = 0
    max_sgn_in_batch = max(max_sgn_in_batch, len(new.sgn))
    max_gls_in_batch = max(max_gls_in_batch, len(new.gls))
    max_txt_in_batch = max(max_txt_in_batch, len(new.txt) + 2)
    sgn_elements = count * max_sgn_in_batch
    gls_elements = count * max_gls_in_batch
    txt_elements = count * max_txt_in_batch
    return max(sgn_elements, gls_elements, txt_elements)


def make_data_iter(
    dataset: Dataset,
    batch_size: int,
    batch_type: str = "sentence",
    train: bool = False,
    shuffle: bool = False,
) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing sgn and optionally txt
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False,
            sort=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=True,
            sort_within_batch=True,
            sort_key=lambda x: len(x.sgn),
            shuffle=shuffle,
        )
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=False,
            sort=False,
        )

    return data_iter
