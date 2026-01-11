import math
import torch
import numpy as np
from torch import Tensor
from torch.nn.functional import interpolate

from .interpolate import interpolate as interpolate_frames


class CenterToKeypoint(torch.nn.Module):
    def __init__(self, center_keypoint: int = 0):
        """
        Args:
            center_keypoint: An integer representing the index of the keypoint to center the pose.
        """
        super(CenterToKeypoint, self).__init__()
        self.center_keypoint = center_keypoint

    def forward(self, pose: Tensor) -> Tensor:
        """
        Normalizes the keypoints of a pose tensor to the difference between those keypoints and a center keypoint.
        Args:
            pose: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        Returns:
            A tensor of shape (frames, people, keypoints, dimensions) representing the normalized pose.
        """
        zero_keypoint = pose[:, :, self.center_keypoint, :].unsqueeze(2)
        return pose + 0.5 - zero_keypoint

    def __str__(self):
        return "CenterToKeypoint"


class NormalizeDistances(torch.nn.Module):
    def __init__(
        self, indices: "tuple[int, int]" = (11, 12), distance_factor: float = 0.2
    ):
        """
        Args:
            indices: A tuple of two integers representing the indices of the keypoints to calculate the distance.
            distance_factor: A float representing the factor to multiply the distance between the keypoints.
        """
        super(NormalizeDistances, self).__init__()
        self.indices = indices
        self.distance_factor = distance_factor

    def forward(self, pose: Tensor) -> Tensor:
        """
        Normalizes the keypoints so the distance between two given keypoints is of a fixed value.
        Args:
            pose: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        Returns:
            A tensor of shape (frames, people, keypoints, dimensions) representing the normalized pose.
        """
        normalized_tensor = pose.clone().detach()
        x1 = pose[:, :, self.indices[0], 0].unsqueeze(2)
        y1 = pose[:, :, self.indices[0], 1].unsqueeze(2)
        x2 = pose[:, :, self.indices[1], 0].unsqueeze(2)
        y2 = pose[:, :, self.indices[1], 1].unsqueeze(2)
        factor = self.distance_factor / torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        normalized_tensor[:, :, :, 0] = x1 - factor * (x1 - pose[:, :, :, 0])
        normalized_tensor[:, :, :, 1] = y1 - factor * (y1 - pose[:, :, :, 1])
        return normalized_tensor

    def __str__(self):
        return "NormalizeDistances"


class FillMissing(torch.nn.Module):
    def forward(self, tensor: Tensor) -> Tensor:
        """
        Fills missing keypoints with the interpolated values between the previous frame where the keypoint was present and the next frame where the keypoint was present.
        If the keypoint is missing in the first frame, it is filled with the next frame where the keypoint is present.
        If the keypoint is missing in the last frame, it is filled with the previous frame where the keypoint is present.
        Args:
            tensor: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        Returns:
            A tensor of shape (frames, people, keypoints, dimensions) representing the pose with missing keypoints filled.
        """
        # rewritten to use numpy interpolate instead of loops and without the need of the prev_valid and next_valid functions
        frames, people, keypoints, dimensions = tensor.size()
        np_tensor = (
            tensor.permute(1, 2, 3, 0)
            .reshape(people * keypoints * dimensions, frames)
            .numpy()
        )
        nan_mask = np.isnan(np_tensor)
        indices = np.arange(np_tensor.shape[1])
        for i in range(np_tensor.shape[0]):
            if np.all(nan_mask[i]):
                np_tensor[i][nan_mask[i]] = 0
                continue
            np_tensor[i][nan_mask[i]] = np.interp(
                indices[nan_mask[i]], indices[~nan_mask[i]], np_tensor[i][~nan_mask[i]]
            )
        tensor = (
            torch.tensor(np_tensor)
            .reshape(people, keypoints, dimensions, frames)
            .permute(3, 0, 1, 2)
        )
        return tensor

    def __str__(self):
        return "FillMissing"


class InterpolateFrames(torch.nn.Module):
    def __init__(self, max_frames: int, mode: str = "linear"):
        """
        Args:
            max_frames: An integer representing the number of frames to interpolate the tensor.
        """
        super(InterpolateFrames, self).__init__()
        self.max_frames = max_frames
        self.mode = mode

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Interpolates the keypoints of a pose tensor to a fixed number of frames.
        Args:
            tensor: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        Returns:
            A tensor of shape (max_frames, people, keypoints, dimensions) representing the interpolated pose.
        """
        frames, people, keypoints, dimensions = tensor.shape
        tensor = interpolate(
            tensor.permute(1, 2, 3, 0).reshape(people, keypoints * dimensions, frames),
            size=self.max_frames,
            mode=self.mode,
        )
        return tensor.reshape(people, keypoints, dimensions, self.max_frames).permute(
            3, 0, 1, 2
        )

    def __str__(self):
        return "InterpolateFrames"


class NormalizeFramesSpeed(torch.nn.Module):
    def __init__(self, max_frames: int, use_faces: bool = False):
        """
        Args:
            max_frames: An integer representing the number of frames to normalize the tensor.
            use_faces: A boolean representing whether to use the face keypoints to compute the amount of movement.
        """
        super(NormalizeFramesSpeed, self).__init__()
        self.max_frames = max_frames
        self.use_faces = use_faces

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Normalizes the keypoints of a pose tensor to the speed of the movement.
        Args:
            tensor: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        """
        frames, people, keypoints, dimensions = tensor.shape

        # compute amount of movement per frame
        tensor_no_nans = torch.nan_to_num(tensor, nan=0.0)
        if not self.use_faces:
            tensor_no_nans = torch.cat(
                [tensor_no_nans[:, :, :32, :], tensor_no_nans[:, :, 500:, :]], dim=2
            )
        movements: list[float] = [
            sum(
                torch.abs(
                    tensor_no_nans[frame_idx, person_idx, :, 0]
                    - tensor_no_nans[frame_idx - 1, person_idx, :, 0]
                )
                .sum()
                .item()
                + torch.abs(
                    tensor_no_nans[frame_idx, person_idx, :, 1]
                    - tensor_no_nans[frame_idx - 1, person_idx, :, 1]
                )
                .sum()
                .item()
                for person_idx in range(people)
            )
            for frame_idx in range(1, frames)
        ]

        # compute the indices to normalize the tensor
        movement_per_frame: float = sum(movements) / (self.max_frames - 1)
        movement_cum = [0.0] + [sum(movements[: i + 1]) for i in range(len(movements))]
        normalized_indices: list[float] = []
        assert (movement_per_frame * 0 == movement_cum[0]) and (
            round(movement_per_frame * (self.max_frames - 1), 5)
            == round(movement_cum[-1], 5)
        ), "First and last frame should have 0 and total movement, respectively"
        for i in range(self.max_frames):
            target_mv = movement_per_frame * i
            j = 0
            while len(normalized_indices) < (i + 1) and j < len(movement_cum):
                if round(movement_cum[j], 5) == round(target_mv, 5):
                    normalized_indices.append(j)
                if round(movement_cum[j], 5) > round(target_mv, 5):
                    normalized_indices.append(
                        j - 1 + (target_mv - movement_cum[j - 1]) / movements[j - 1]
                    )
                j += 1

        # for each index, get the frame from the tensor or interpolate between two frames if the index is not an integer
        normalized_tensor = []
        for idx in normalized_indices:
            if type(idx) == int:
                normalized_tensor.append(tensor[int(idx)])
            else:
                frame_1 = tensor[int(math.floor(idx))]
                frame_2 = tensor[int(math.ceil(idx))]
                factor = idx - int(math.floor(idx))
                interpolated_frame = interpolate_frames(frame_1, frame_2, factor)
                normalized_tensor.append(interpolated_frame.clone().detach())

        return torch.stack(normalized_tensor)

    def __str__(self):
        return "NormalizeFramesSpeed"


class FilterLandmarks(torch.nn.Module):
    def __init__(self, mask: Tensor, use_3d: bool = False):
        """
        Args:
            mask: tensor of shape (L,) where L is the number of landmarks.
            use_3d: boolean indicating whether to use the 3D coordinates.
        """
        super(FilterLandmarks, self).__init__()
        self.mask = mask
        self.use_3d = use_3d

    def forward(self, pose: Tensor) -> Tensor:
        """
        Filter landmarks by mask.
        Args:
            pose: tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        Returns:
            tensor of same (frames, people, keypoints', dimensions') with landmarks filtered by mask.
        """
        # transpose to (L, S, P, D) for filtering
        pose = pose.permute(2, 1, 0, 3)
        pose = pose[self.mask]
        if not self.use_3d:
            pose = pose[:, :, :, :2]
        return pose.permute(2, 1, 0, 3)

    def __str__(self):
        return "FilterLandmarks"


class PadTruncateFrames(torch.nn.Module):
    def __init__(self, max_len: int):
        """
        Args:
            max_len: An integer representing the maximum length to pad or truncate the tensor.
        """
        super(PadTruncateFrames, self).__init__()
        self.max_len = max_len

    def forward(self, datum: Tensor) -> Tensor:
        """
        Pads or truncates the pose tensor to a fixed length.
        Args:
            datum: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        """
        frames, people, keypoints, dimensions = datum.shape
        if frames < self.max_len:
            return torch.cat(
                [
                    datum,
                    torch.zeros(self.max_len - frames, people, keypoints, dimensions),
                ]
            )
        else:
            return datum[: self.max_len]

    def __str__(self):
        return "PadTruncateFrames"


class RandomSampleFrames(torch.nn.Module):
    def __init__(self, rate: int):
        """
        Args:
            rate: An integer representing the rate to sample the tensor.
        """
        super(RandomSampleFrames, self).__init__()
        self.rate = rate

    def forward(self, pose: Tensor) -> Tensor:
        """
        Randomly samples 1 frame per rate frames.
        Args:
            pose: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        Returns:
            A tensor of shape (frames // rate, people, keypoints, dimensions) where each frame is sampled randomly but following a sequence.
        """
        frames, people, keypoints, dimensions = pose.shape
        chunks_indices = torch.arange(0, frames, self.rate)
        random_indices = torch.randint(0, self.rate, (math.ceil(frames / self.rate),))
        random_indices += chunks_indices
        # if last index is greater than the number of frames, set it to the last frame
        if random_indices[-1] >= frames:
            random_indices[-1] = frames - 1
        return pose[random_indices]

    def __str__(self):
        return "RandomSampleFrames"


class RandomSampleFrameLegacy(torch.nn.Module):
    def __init__(self, max_len: int):
        """
        Args:
            max_len: An integer representing the maximum length to sample the frames.
        """
        super(RandomSampleFrameLegacy, self).__init__()
        self.max_len = max_len

    def forward(self, pose: Tensor) -> Tensor:
        """
        Randomly samples 1 frame per max_len chunks of frames.
        Args:
            pose: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        """
        if pose.size(0) < self.max_len:
            return torch.cat(
                [
                    pose,
                    torch.zeros(
                        self.max_len - pose.size(0),
                        pose.size(1),
                        pose.size(2),
                        pose.size(3),
                    ),
                ]
            )
        indices = []
        chunk_size = pose.size(0) // self.max_len
        for i in range(0, self.max_len):
            indices.append(torch.randint(i * chunk_size, (i + 1) * chunk_size, [1]))
        return pose[indices, :, :, :]

    def __str__(self):
        return "RandomSampleFrameLegacy"


class ReplaceNansWithZeros(torch.nn.Module):
    def forward(self, tensor: Tensor) -> Tensor:
        """
        Replaces NaN values in the tensor with zeros.
        Args:
            tensor: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        """
        return torch.nan_to_num(tensor, nan=0.0)

    def __str__(self):
        return "ReplaceNansWithZeros"


class UseFramesDiffs(torch.nn.Module):
    def forward(self, tensor: Tensor) -> Tensor:
        """
        Represents frames as the difference between consecutive frames.
        Args:
            tensor: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        Returns:
                        A tensor of shape (frames - 1, people, keypoints, dimensions) representing the difference between consecutive frames.
        """
        return tensor[1:] - tensor[:-1]

    def __str__(self):
        return "UseFramesDiffs"


class FlattenKeypoints(torch.nn.Module):
    def forward(self, tensor: Tensor) -> Tensor:
        """
        Reshape the pose of datum only keeping the first dimension S (sequence lenght) and flattening the number of landmarks L and their dimensions D.
        Args:
                datum: Tensor of shape (S, P, D, L)
        Returns:
                Tensor of shape (frames, P * D * L)
        """
        frames, people, keypoints, dimensions = tensor.shape
        return tensor.view(frames, people * keypoints * dimensions)

    def __str__(self):
        return "FlattenKeypoints"
