import argparse
import json
import math
import os
import random
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm


def read_content4all_json(json_path: str) -> pd.DataFrame:
    """Read Content4All annotations JSON and return a row-wise DataFrame.

    The input JSON is expected to follow the structure used in
    dataset_formatting/content4all/gen_annotations.ipynb, where
    pd.DataFrame(json_obj).T yields rows like:
        sequence_id, video_id, video_path, start_ms, stop_ms, start_frame,
        stop_frame, openpose2D_path, openpose-2Dto3D_path, annotation
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data).T
    # Normalize types
    df["sequence_id"] = df["sequence_id"].astype(str)
    df["video_id"] = df["video_id"].astype(str)
    # Build id
    df["id"] = df["video_id"] + "_" + df["sequence_id"]
    # Rename annotation to text
    if "annotation" in df.columns:
        df.rename(columns={"annotation": "text"}, inplace=True)
    return df


def stratified_group_splits(
    df: pd.DataFrame,
    group_col: str,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> pd.Series:
    """Create non-overlapping splits by grouping on group_col.

    Ensures that all rows sharing the same group are assigned to the same split.
    Ratios are approximate by number of groups.
    """
    assert 0 < test_ratio < 1 and 0 < val_ratio < 1 and test_ratio + val_ratio < 1
    rng = random.Random(seed)
    groups = sorted(df[group_col].astype(str).unique())
    rng.shuffle(groups)

    num_groups = len(groups)
    num_test = int(round(num_groups * test_ratio))
    num_val = int(round(num_groups * val_ratio))
    num_train = max(0, num_groups - num_test - num_val)

    test_groups = set(groups[:num_test])
    val_groups = set(groups[num_test : num_test + num_val])
    train_groups = set(groups[num_test + num_val : num_test + num_val + num_train])

    def assign(group: str) -> str:
        if group in test_groups:
            return "test"
        if group in val_groups:
            return "val"
        return "train"

    return df[group_col].astype(str).map(assign)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def seconds_from_ms(ms: float) -> float:
    return float(ms) / 1000.0


def build_ffmpeg_cmd(src: str, dst: str, start_s: float, end_s: float, reencode: bool) -> List[str]:
    # Use re-encode for higher precision cuts; copy is faster but can be imprecise
    duration = max(0.0, end_s - start_s)
    if duration <= 0:
        # Degenerate; still run to create a tiny clip of 0.01s
        duration = 0.01

    if reencode:
        return [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.3f}",
            "-i",
            src,
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            dst,
        ]
    else:
        return [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.3f}",
            "-i",
            src,
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            dst,
        ]


def extract_clip(
    src_video: str,
    dst_video: str,
    start_ms: float,
    stop_ms: float,
    reencode: bool,
) -> Tuple[str, bool, str]:
    try:
        ensure_dir(os.path.dirname(dst_video))
        start_s = seconds_from_ms(start_ms)
        end_s = seconds_from_ms(stop_ms)
        cmd = build_ffmpeg_cmd(src_video, dst_video, start_s, end_s, reencode)
        subprocess.run(cmd, check=True)
        return (dst_video, True, "")
    except subprocess.CalledProcessError as e:
        return (dst_video, False, f"ffmpeg failed: {e}")
    except Exception as e:
        return (dst_video, False, str(e))


def write_metadata(output_dir: str, source_root: str) -> None:
    metadata: Dict[str, object] = {
        "name": "Content4All VRT News",
        "id": "Content4All-VRT-NEWS",
        "url": None,
        "download_link": None,
        "mirror_link": source_root,
        "input_language": "VGT",
        "output_language": "nl",
        "input_types": ["video"],
        "output_types": ["text"],
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Format Content4All VRT-NEWS into SLTDataset layout"
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to Content4All annotations JSON (vrt-news-annotations.json)",
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Root directory where relative video paths (Videos/...) resolve",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Target SLTDataset directory (will contain annotations.csv, videos/)",
    )
    parser.add_argument(
        "--extract-videos",
        action="store_true",
        help="If set, extracts per-segment videos to videos/{id}.mp4",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="If set, re-encode segments (more precise, slower). Default copies.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Parallel workers for video extraction",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of groups for test split",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of groups for validation split",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splits"
    )

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    videos_dir = os.path.join(output_dir, "videos")
    ensure_dir(output_dir)
    ensure_dir(videos_dir)

    # 1) Read source JSON and build annotations
    df = read_content4all_json(args.json)

    # 2) Build split by grouping on video_id
    if "video_id" not in df.columns:
        raise RuntimeError("Expected 'video_id' in annotations JSON")
    df["split"] = stratified_group_splits(
        df, group_col="video_id", test_ratio=args.test_ratio, val_ratio=args.val_ratio, seed=args.seed
    )

    # 3) Columns for annotations.csv
    # Keep extended info; SLTDataset uses id, text, split minimally
    cols = [
        "id",
        "text",
        "split",
        "video_id",
        "video_path",
        "start_ms",
        "stop_ms",
        "start_frame",
        "stop_frame",
        "openpose2D_path",
        "openpose-2Dto3D_path",
    ]
    existing_cols = [c for c in cols if c in df.columns]
    annotations = df[existing_cols].copy()

    # 4) Write metadata and annotations.csv
    write_metadata(output_dir, os.path.abspath(args.source_root))
    annotations_path = os.path.join(output_dir, "annotations.csv")
    annotations.to_csv(annotations_path, index=False)

    # 5) Optional: extract per-segment videos
    if args.extract_videos:
        missing_sources = 0
        futures = []
        with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
            for _, row in annotations.iterrows():
                rel = row["video_path"] if "video_path" in row else None
                if not rel or not isinstance(rel, str):
                    continue
                src_video = os.path.join(args.source_root, rel)
                if not os.path.exists(src_video):
                    missing_sources += 1
                    continue
                dst_video = os.path.join(videos_dir, f"{row['id']}.mp4")
                start_ms = float(row["start_ms"]) if "start_ms" in row else 0.0
                stop_ms = float(row["stop_ms"]) if "stop_ms" in row else start_ms
                futures.append(
                    ex.submit(
                        extract_clip,
                        src_video,
                        dst_video,
                        start_ms,
                        stop_ms,
                        args.reencode,
                    )
                )

            results = []
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting videos"):
                results.append(fut.result())

        failed = [r for r in results if not r[1]]
        print(
            f"Video extraction done. Total: {len(results)}, failed: {len(failed)}, missing sources: {missing_sources}"
        )
        if failed:
            print("Some extractions failed (showing up to first 20):")
            for path, _, err in failed[:20]:
                print(f" - {path}: {err}")

    print(f"Wrote {annotations.shape[0]} annotations to {annotations_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()








