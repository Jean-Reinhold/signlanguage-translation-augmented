from typing import Optional

import torch
from torch import Tensor


def interpolate(frame_1: Tensor, frame_2: Tensor, factor: float) -> Tensor:
    """
    Interpolates between two frames of keypoints.
    Args:
        frame_1: A tensor of shape (people, keypoints, dimensions) representing the keypoints of the first frame.
        frame_2: A tensor of shape (people, keypoints, dimensions) representing the keypoints of the second frame.
        factor: A float representing the factor to interpolate between the frames (0.0 to 1.0).
    Returns:
        A tensor of shape (people, keypoints, dimensions) representing the interpolated keypoints.
    """
    assert 0.0 <= factor <= 1.0, "Interpolation factor must be between 0.0 and 1.0"
    return frame_1 * factor + frame_2 * (1 - factor)


def prev_valid(tensor: Tensor, frame: int, person: int, keypoint: int) -> Optional[int]:
    """
    Returns the previous frame where the keypoint was present. If no previous frame is found, returns None.
    Args:
        tensor: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        frame: An integer representing the current frame.
        person: An integer representing the current person.
        keypoint: An integer representing the current keypoint.
    Returns:
        An integer representing the previous frame where the keypoint was present or None if no valid previous frame is found.
    """
    while torch.isnan(tensor[frame, person, keypoint, 0]):
        frame -= 1
        if frame < 0:
            return None
    return frame


def next_valid(tensor: Tensor, frame: int, person: int, keypoint: int) -> Optional[int]:
    """
    Returns the next frame where the keypoint was present. If no next frame is found, returns None.
    Args:
        tensor: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        frame: An integer representing the current frame.
        person: An integer representing the current person.
        keypoint: An integer representing the current keypoint.
    Returns:
        An integer representing the next frame where the keypoint was present or None if no valid next frame is found.
    """
    while torch.isnan(tensor[frame, person, keypoint, 0]):
        frame += 1
        if frame >= tensor.shape[0]:
            return None
    return frame
