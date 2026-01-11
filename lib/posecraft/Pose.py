from typing import Optional, Literal
from textwrap import wrap

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


Component = Literal["body", "face", "lhand", "rhand"]


class Pose:
    """
    Pose class to store and manipulate pose data.
    Each pose is a tensor of shape (F, P, K, D) where:
        - F is the number of frames
        - P is the number of people
        - K is the number of keypoints
        - D is the number of dimensions (2 for 2D, 3 for 3D).
    """

    @staticmethod
    def load_to_tensor(path: str) -> torch.Tensor:
        return torch.from_numpy(np.load(path)).float()

    def __init__(self, path: Optional[str] = None, pose: Optional[Tensor] = None):
        """
        Initialize a Pose object from a npy file or a Pytorch tensor.
        Args:
            path (str): Path to the pose tensor file.
            pose (Tensor): Pose tensor.
        """
        if pose is not None:
            self.pose = pose
        elif path is not None:
            self.pose = self.load_to_tensor(path)
        assert self.pose.dim() == 4, "Pose tensor must have 4 dimensions"
        self.F = self.pose.shape[0]
        self.P = self.pose.shape[1]
        self.K = self.pose.shape[2]
        self.D = self.pose.shape[3]

    COMPONENT_MAP = (
        ["body" for _ in range(33)]
        + ["face" for _ in range(468)]
        + ["lhand" for _ in range(21)]
        + ["rhand" for _ in range(21)]
    )

    @staticmethod
    def get_components_mask(
        include: list[Component] = [], exclude: list[Component] = []
    ) -> Tensor:
        """
        Get a mask to filter keypoints based on their component.
        Args:
            include (list[Component]): List of components to include.
            exclude (list[Component]): List of components to exclude.
        Returns:
            Tensor: Boolean mask to filter keypoints.
        """
        assert not (
            include and exclude
        ), "Only one of include or exclude should be provided"
        if include:
            return Tensor(
                [True if kp in include else False for kp in Pose.COMPONENT_MAP]
            ).bool()
        else:
            return Tensor(
                [False if kp in exclude else True for kp in Pose.COMPONENT_MAP]
            ).bool()

    def animate(
        self,
        h: Optional[int] = 4,
        w: Optional[int] = 4,
        size=10,
        person_idx=0,
        title: Optional[str] = None,
    ):
        def get_update(scatter):
            def update(frame):
                # Reshape keypoints to have each pair (x, y) in a row
                x = self.pose[frame, person_idx, :, 0]
                y = self.pose[frame, person_idx, :, 1]
                scatter.set_offsets(torch.stack((x, y), dim=-1))
                return (scatter,)

            return update

        fig, ax = plt.subplots(figsize=(h, w))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()  # Invert y-axis so origin is at top-left
        scatter = ax.scatter([], [], s=size)
        if title:
            ax.set_title("\n".join(wrap(title, 60)), fontsize=6)
        return FuncAnimation(
            fig, get_update(scatter), frames=self.F, interval=50, blit=True
        )
