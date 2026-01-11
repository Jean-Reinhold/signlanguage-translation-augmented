import os
import sys
import json
from typing import Literal, Optional, TypedDict

# Add lib directory to path for local imports
_lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "lib")
if _lib_path not in sys.path:
    sys.path.insert(0, _lib_path)

import pandas as pd
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose

from posecraft.Pose import Pose
from .WordLevelTokenizer import WordLevelTokenizer

# patch numpy types to solve skvideo issue: https://github.com/scikit-video/scikit-video/issues/154#issuecomment-1445239790
import numpy

numpy.float = numpy.float64  # type: ignore
numpy.int = numpy.int_  # type: ignore
from skvideo.io import vread, vwrite  # type: ignore


InputType = Literal["video", "pose"]
OutputType = Literal["text", "gloss"]


class Metadata(TypedDict):
    name: str
    id: str
    url: str
    download_link: Optional[str]
    mirror_link: Optional[str]
    input_language: str
    output_language: str
    input_types: list[InputType]
    output_types: list[OutputType]


class SLTDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        input_mode: InputType,
        output_mode: OutputType,
        split: Optional[Literal["train", "val", "test"]] = None,
        transforms: Optional[Compose] = None,
        tokenizer=WordLevelTokenizer(),
        max_tokens: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.split = split
        self.transforms = transforms
        self.tokenizer = tokenizer

        try:
            self.metadata: Metadata = json.load(
                open(os.path.join(data_dir, "metadata.json"))
            )
            print(f"Loaded metadata for dataset: {self.metadata['name']}")
            self.annotations = pd.read_csv(
                os.path.join(data_dir, os.path.join(data_dir, "annotations.csv"))
            )

            if self.split is not None:
                assert (
                    "split" in self.annotations.columns
                ), "Split annotations not found"
                self.annotations["split"] = self.annotations["split"].astype(str)
            if self.output_mode == "text":
                assert "text" in self.annotations.columns, "Text annotations not found"
                self.annotations["text"] = self.annotations["text"].astype(str)
            elif self.output_mode == "gloss":
                assert (
                    "gloss" in self.annotations.columns
                ), "Gloss annotations not found"
                self.annotations["gloss"] = self.annotations["gloss"].astype(str)

            # filter samples above max_tokens by splitting output mode by whitespace
            if max_tokens is not None:
                self.annotations = self.annotations[
                    self.annotations[self.output_mode].str.split().apply(len)
                    <= max_tokens
                ]
                self.annotations.reset_index(drop=True, inplace=True)

            # initialize tokenizer on train split if available, else on all data
            target_data = (
                self.annotations[self.output_mode].tolist()
                if split is None
                else self.annotations[self.annotations["split"] == "train"][
                    self.output_mode
                ].tolist()
            )
            self.tokenizer.fit(target_data)
            if split is not None:
                self.annotations = self.annotations[self.annotations["split"] == split]
                self.annotations.reset_index(drop=True, inplace=True)
            print(
                f"Loaded {split if split is not None else ''} annotations at {os.path.join(data_dir, 'annotations.csv')}"
            )
        except FileNotFoundError:
            raise FileNotFoundError("Metadata or annotations not found")

        self.token_ids: Tensor = self.tokenizer(
            self.annotations[output_mode].tolist(),
            padding=("max_length" if max_tokens is not None else "longest"),
            max_length=max_tokens,
            return_tensors="pt",
        )  # type: ignore

        self.missing_files = []
        for id in tqdm(self.annotations["id"], desc="Validating files"):
            path = (
                os.path.join(self.data_dir, "poses", f"{id}.npy")
                if self.input_mode == "pose"
                else os.path.join(self.data_dir, "videos", f"{id}.mp4")
            )
            if not os.path.exists(path):
                self.missing_files.append(id)
        if len(self.missing_files) > 0:
            print(
                f"Missing {len(self.missing_files)} files out of {len(self.annotations)} ({round(100 * len(self.missing_files) / len(self.annotations), 2)}%)"
                + (f" from split {split}" if split is not None else "")
            )
            # remove missing files
            self.annotations = self.annotations[
                ~self.annotations["id"].isin(self.missing_files)
            ]
        else:
            print("Dataset loaded correctly")
        print()

    def __len__(self) -> int:
        return len(self.annotations)

    def get_pose(self, idx: int) -> Tensor:
        id = self.annotations.iloc[idx]["id"]
        file_path = os.path.join(self.data_dir, "poses", f"{id}.npy")
        return Pose.load_to_tensor(file_path)

    def get_video(self, idx: int) -> Tensor:
        id = self.annotations.iloc[idx]["id"]
        return torch.from_numpy(
            vread(os.path.join(self.data_dir, "videos", f"{id}.mp4"))
        )

    def get_text(self, idx: int) -> str:
        return self.annotations.iloc[idx]["text"]

    def get_gloss(self, idx: int) -> str:
        return self.annotations.iloc[idx]["gloss"]

    def get_item_raw(self, idx: int) -> tuple[Tensor, str]:
        if self.input_mode == "pose":
            x_data = self.get_pose(idx)
        elif self.input_mode == "video":
            x_data = self.get_video(idx)
        if self.output_mode == "text":
            y_data = self.get_text(idx)
        elif self.output_mode == "gloss":
            y_data = self.get_gloss(idx)
        return x_data, y_data

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x_data, y_data = self.get_item_raw(idx)
        if self.transforms:
            x_data = self.transforms(x_data)
        y_data = self.token_ids[idx]
        return x_data, y_data

    def visualize_pose(
        self,
        idx: int,
        h: Optional[int] = 4,
        w: Optional[int] = 4,
        size=10,
        transforms: Optional[Compose] = None,
    ):
        pose, text = self.get_item_raw(idx)
        pose = Pose(pose=transforms(pose) if transforms else pose)
        anim = pose.animate(h=h, w=w, size=size, title=text)
        return anim
