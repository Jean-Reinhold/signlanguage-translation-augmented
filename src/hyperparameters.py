import json
import sys
import os
from typing import TypedDict, Optional

import torch

# Add lib directory to path for local imports
_lib_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib")
if _lib_path not in sys.path:
    sys.path.insert(0, _lib_path)

from slt_datasets.SLTDataset import InputType, OutputType
from posecraft.Pose import Component, Pose
from posecraft.transforms import (
    CenterToKeypoint,
    NormalizeDistances,
    FillMissing,
    InterpolateFrames,
    NormalizeFramesSpeed,
    FilterLandmarks,
    PadTruncateFrames,
    RandomSampleFrames,
    RandomSampleFrameLegacy,
    ReplaceNansWithZeros,
    UseFramesDiffs,
    FlattenKeypoints,
)


class HyperParameters(TypedDict):
    INPUT_MODE: InputType
    OUTPUT_MODE: OutputType
    BATCH_SIZE: int
    MAX_FRAMES: int
    MAX_TOKENS: int
    SAMPLE_RATE: Optional[int]
    LANDMARKS_USED: list[Component]
    USE_3D: bool
    TRANSFORMS: list[torch.nn.Module]
    D_MODEL: int
    DROPOUT: float
    NUM_ENCODER_LAYERS: int
    NUM_DECODER_LAYERS: int
    USE_CLASS_WEIGHTS: bool
    LR: float


def load_hyperparameters_from_json(path: str) -> HyperParameters:
    with open(path, "r") as f:
        hp = json.load(f)
    transforms = []
    for transform in hp["TRANSFORMS"]:
        if transform == "CenterToKeypoint":
            transforms.append(CenterToKeypoint())
        elif transform == "NormalizeDistances":
            transforms.append(NormalizeDistances())
        elif transform == "FillMissing":
            transforms.append(FillMissing())
        elif transform == "InterpolateFrames":
            transforms.append(InterpolateFrames(hp["MAX_FRAMES"]))
        elif transform == "NormalizeFramesSpeed":
            transforms.append(NormalizeFramesSpeed(hp["MAX_FRAMES"]))
        elif transform == "FilterLandmarks":
            transforms.append(
                FilterLandmarks(
                    Pose.get_components_mask(hp["LANDMARKS_USED"]), hp["USE_3D"]
                )
            )
        elif transform == "PadTruncateFrames":
            transforms.append(PadTruncateFrames(hp["MAX_FRAMES"]))
        elif transform == "RandomSampleFrames":
            transforms.append(
                RandomSampleFrames(hp["SAMPLE_RATE"] if "SAMPLE_RATE" in hp else 1)
            )
        elif transform == "RandomSampleFrameLegacy":
            transforms.append(RandomSampleFrameLegacy(hp["MAX_FRAMES"]))
        elif transform == "ReplaceNansWithZeros":
            transforms.append(ReplaceNansWithZeros())
        elif transform == "UseFramesDiffs":
            transforms.append(UseFramesDiffs())
        elif transform == "FlattenKeypoints":
            transforms.append(FlattenKeypoints())
    hp["TRANSFORMS"] = transforms
    return hp
