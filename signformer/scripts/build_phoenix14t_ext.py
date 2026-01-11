#!/usr/bin/env python3
import argparse
import gzip
import logging
import os
import pickle
from typing import Dict, List, Tuple

import pandas as pd


def load_pickle_list(path: str) -> List[dict]:
    with gzip.open(path, "rb") as f:
        obj = pickle.load(f)
        return obj if isinstance(obj, list) else [obj]


def write_pickle_list(path: str, samples: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_indices(df: pd.DataFrame) -> Tuple[Dict[str, pd.core.groupby.generic.DataFrameGroupBy], Dict[str, pd.DataFrame]]:
    # Normalize columns
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["split"] = df["split"].astype(str)
    df["text_norm"] = df["text"].astype(str).str.strip()
    if "gloss" in df.columns:
        df["gloss"] = df["gloss"].astype(str)

    by_split = {s: g for s, g in df.groupby("split")}
    by_split_id = {s: g.groupby("id") for s, g in by_split.items()}
    by_split_text = {s: g.set_index("text_norm") for s, g in by_split.items()}
    return by_split_id, by_split_text


def make_ext_for_split(
    split_name: str,
    in_path: str,
    out_path: str,
    by_split_id: Dict[str, pd.core.groupby.generic.DataFrameGroupBy],
    by_split_text: Dict[str, pd.DataFrame],
    logger: logging.Logger,
) -> Tuple[int, int, int, int]:
    base_samples = load_pickle_list(in_path)
    out_samples: List[dict] = []
    grp_by_id = by_split_id.get(split_name)
    text_index = by_split_text.get(split_name)

    n_in = len(base_samples)
    n_id_match = 0
    n_text_fallback = 0
    n_no_match = 0

    for sample in base_samples:
        name = sample.get("name", "")
        if "/" in name:
            _, id_name = name.split("/", 1)
        else:
            id_name = name

        rows = None
        if grp_by_id is not None and id_name in grp_by_id.groups:
            rows = grp_by_id.get_group(id_name)
            n_id_match += 1
        else:
            txt = str(sample.get("text", "")).strip()
            if text_index is not None and txt in text_index.index:
                loc = text_index.loc[txt]
                rows = loc.to_frame().T if isinstance(loc, pd.Series) else loc
                n_text_fallback += 1
            else:
                # Keep original sample if no augmentation row found
                out_samples.append(sample)
                n_no_match += 1
                continue

        if isinstance(rows, pd.Series):
            rows = rows.to_frame().T

        for _, r in rows.iterrows():
            dup = dict(sample)
            dup["text"] = str(r["text"]).strip()
            if "gloss" in r:
                dup["gloss"] = str(r["gloss"]).strip()
            if "signer" in r:
                dup["signer"] = str(r["signer"]).strip()
            out_samples.append(dup)

    write_pickle_list(out_path, out_samples)
    logger.info(
        "%s in=%d out=%d id_matches=%d text_fallbacks=%d no_match_kept=%d",
        split_name,
        n_in,
        len(out_samples),
        n_id_match,
        n_text_fallback,
        n_no_match,
    )
    return n_in, len(out_samples), n_id_match, n_text_fallback


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build phoenix14t-ext pickles by duplicating samples per augmented texts")
    ap.add_argument("--data_dir", type=str, default="PHOENIX2014T", help="Directory containing original pickles and TSV")
    ap.add_argument("--tsv", type=str, default="train_aug.tsv", help="TSV file with columns: id, text, gloss, signer, split")
    ap.add_argument("--in_train", type=str, default="phoenix14t.pami0.train")
    ap.add_argument("--in_dev", type=str, default="phoenix14t.pami0.dev")
    ap.add_argument("--in_test", type=str, default="phoenix14t.pami0.test")
    ap.add_argument("--out_prefix", type=str, default="phoenix14t-ext.pami0")
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("build_phoenix14t_ext")

    args = parse_args()
    tsv_path = args.tsv if os.path.isabs(args.tsv) else os.path.join(args.data_dir, args.tsv)
    df = pd.read_csv(tsv_path, sep="\t")
    by_split_id, by_split_text = build_indices(df)

    in_train = args.in_train if os.path.isabs(args.in_train) else os.path.join(args.data_dir, args.in_train)
    in_dev = args.in_dev if os.path.isabs(args.in_dev) else os.path.join(args.data_dir, args.in_dev)
    in_test = args.in_test if os.path.isabs(args.in_test) else os.path.join(args.data_dir, args.in_test)

    out_train = os.path.join(args.data_dir, f"{args.out_prefix}.train")
    out_dev = os.path.join(args.data_dir, f"{args.out_prefix}.dev")
    out_test = os.path.join(args.data_dir, f"{args.out_prefix}.test")

    make_ext_for_split("train", in_train, out_train, by_split_id, by_split_text, logger)
    make_ext_for_split("dev", in_dev, out_dev, by_split_id, by_split_text, logger)
    make_ext_for_split("test", in_test, out_test, by_split_id, by_split_text, logger)


if __name__ == "__main__":
    main()


