from typing import Literal, Optional, Callable

import torch
from torch import Tensor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from hyperparameters import HyperParameters


def plot_encoder_layers(
    attn_output_weights: list[Tensor],
    hp: HyperParameters,
    output_path: Optional[str] = None,
    figsize: tuple[int, int] = (20, 20),
    transparent: bool = False,
    src_padding_mask: Optional[Tensor] = None,
):
    """
    Plot the attention weights of the encoder layers.
    Args:
        attn_output_weights (list[Tensor]): List of tensors of shape (B, N, L) where B is the batch size, N is the number of words and L is the number of words in the source sentence.
        hp (HyperParameters): Hyperparameters object.
        output_path (str): Path to save the plot.
        figsize (tuple[int, int]): Size of the figure.
        transparent (bool): Whether to save the plot with a transparent background.
        src_padding_mask (Tensor): Mask of shape (B, L) where L is the number of words in the source sentence.
    Returns:
        fig, ax: Figure and axis objects.
    """
    fig, axes = plt.subplots(1, hp["NUM_ENCODER_LAYERS"], figsize=figsize, sharey=True)
    for layer, attn_weights in enumerate(attn_output_weights):
        ax = (
            axes[layer] if hp["NUM_ENCODER_LAYERS"] > 1 else axes
        )  # Handle case with only one layer
        if src_padding_mask is not None:
            attn_weights = attn_weights[:, ~src_padding_mask]
        sns.heatmap(
            attn_weights,
            ax=ax,
            square=True,
            cbar=False,
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(f"Layer {layer+1}")

    if output_path is not None:
        file_extension = "png" if transparent else "jpg"
        plt.savefig(
            f"{output_path}/attn_self_heatmaps_encoder_layers.{file_extension}",
            dpi=150,
            transparent=transparent,
        )
    return fig, ax


def plot_decoder_layers(
    attn_output_weights: dict[int, list[Tensor]],
    hp: HyperParameters,
    translation: list[str],
    mode: Literal["self", "cross"],
    output_path: Optional[str] = None,
    figsize: tuple[int, int] = (20, 20),
    transparent: bool = False,
    src_padding_mask: Optional[Tensor] = None,
):
    """
    Plot the attention weights of the decoder layers.
    Args:
        attn_output_weights (dict[int, list[Tensor]]): Dictionary of tensors of shape (B, N, L) where B is the batch size, N is the number of words and L is the number of words in the source sentence.
        hp (HyperParameters): Hyperparameters object.
        translation (list[str]): List of words in the translation.
        mode (Literal["self", "cross"]): Whether to plot self-attention or cross-attention.
        output_path (str): Path to save the plot.
        figsize (tuple[int, int]): Size of the figure.
        transparent (bool): Whether to save the plot with a transparent background.
        src_padding_mask (Tensor): Mask of shape (B, L) where L is the number of words in the source sentence.
    Returns:
        fig, ax: Figure and axis objects.
    """
    tgt_length = len(translation) - 1  # from BOS to EOS-1
    fig, axes = plt.subplots(
        tgt_length, hp["NUM_DECODER_LAYERS"], figsize=figsize, sharey=True
    )
    for layer in attn_output_weights:
        for token, attn_weights in enumerate(attn_output_weights[layer]):
            attn_weights = attn_weights[0]
            ax = (
                axes[token, layer] if hp["NUM_DECODER_LAYERS"] > 1 else axes[token]
            )  # Handle case with only one layer
            tgt_sent = translation[1 : attn_weights.shape[0] + 1]
            if mode == "cross" and src_padding_mask is not None:
                attn_weights = attn_weights[:, ~src_padding_mask]
            sns.heatmap(
                attn_weights,
                ax=ax,
                square=True,
                cbar=True,  # mode == "self",
                vmin=0.0 if mode == "self" else None,
                vmax=1.0 if mode == "self" else None,
            )
            ax.set_aspect("auto")
            ax.set_yticklabels(tgt_sent, rotation=0)

            ax.set_ylabel("Predicted token", fontsize=12)
            ax.set_xlabel("Percentage of the video", fontsize=12)

            if mode == "self":
                ax.set_xticklabels(tgt_sent, rotation=90)
            else:
                # FIXME: This is a hack to make the x-axis labels show up correctly
                max_value = attn_weights.shape[1]
                percentages = ["0", "25", "50", "75", "100"]
                num_partitions = len(percentages)
                step = max_value / (num_partitions - 1)
                partitions = [i * step for i in range(num_partitions)]
                ax.set_xticks(partitions, ["0", "25", "50", "75", "100"])
                ax.set_xlim(0, max_value)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_title(f"Layer {layer + 1}") if token == 0 else None
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    if output_path:
        file_extension = "png" if transparent else "jpg"
        plt.savefig(
            f"{output_path}/attn_{mode}_heatmaps_decoder_layers.{file_extension}",
            dpi=150,
            transparent=transparent,
        )
    return fig, axes


def preprocess_attn_weights(
    attn_weights: list[Tensor],
    norm_func: Callable = lambda t: (t - t.min()) / (t.max() - t.min()),
    take_last: bool = False,
):
    """
    Preprocess the attention weights for plotting, taking for each token the attention weights of the last call where the word is the target.
    Args:
        attn_weights (list[Tensor]): List of tensors of shape (B, N, L) where B is the batch size, N is the number of words and L is the number of words in the source sentence.
        norm_func (Callable): Function to normalize the attention weights.
        take_last (bool): If True, take the attention weights of the last call for all tokens. If False, take the attention weights of the call where the word is the target.
    Returns:
        processed_attn_weights: Processed attention weights.
    """
    # take for each word the attention weights of the call where the word is the target
    processed_attn_weights = torch.zeros_like(attn_weights[-1])
    for i, attn_output_weights in enumerate(attn_weights):
        processed_attn_weights[0, i, :] = norm_func(
            attn_output_weights[0, (-1 if take_last else i), :]
        )
    processed_attn_weights = processed_attn_weights.squeeze(0)
    return processed_attn_weights


def plot_decoder_attn_per_frame(
    decoder_ca: dict[int, list[Tensor]],
    hp: HyperParameters,
    mode: Literal["heatmap", "lineplot"],
    translation: list[str],
    output_path: Optional[str] = None,
    transparent: bool = False,
    figsize: tuple[int, int] = (10, 10),
    src_padding_mask: Optional[Tensor] = None,
    norm_func: Callable = lambda t: (t - t.min()) / (t.max() - t.min()),
):
    """
    Plot the attention weights of the decoder layers per frame.
    Args:
        decoder_ca (dict[int, list[Tensor]]): Dictionary of tensors of shape (B, N, L) where B is the batch size, N is the number of words and L is the number of words in the source sentence.
        mode (Literal["heatmap", "lineplot"]): Whether to plot the attention weights as a heatmap or lineplot.
        translation (list[str]): List of words in the translation.
        output_path (str): Path to save the plot.
        transparent (bool): Whether to save the plot with a transparent background.
        figsize (tuple[int, int]): Size of the figure.
        src_padding_mask (Tensor): Mask of shape (B, L) where L is the number of words in the source sentence.
        norm_func (Callable): Function to normalize the attention weights.
    Returns:
        fig, axes: Figure and axis objects
    """
    fig, axes = plt.subplots(len(decoder_ca), 1, figsize=figsize, sharey=True)
    for layer in decoder_ca:
        ax = (
            axes[layer] if hp["NUM_DECODER_LAYERS"] > 1 else axes
        )  # Handle case with only one layer
        attn_weights = preprocess_attn_weights(decoder_ca[layer], norm_func)
        if src_padding_mask is not None:
            attn_weights = attn_weights[:, ~src_padding_mask]
        tgt_sent = translation[1 : attn_weights.shape[0] + 1]
        ax.set_title(f"Layer {layer + 1}")
        if mode == "heatmap":
            sns.heatmap(
                attn_weights,
                yticklabels=tgt_sent,
                ax=ax,
            )
        elif mode == "lineplot":
            df_attn_weights = pd.DataFrame(attn_weights.T.tolist())
            df_attn_weights.columns = tgt_sent
            ax = sns.lineplot(df_attn_weights, dashes=False, ax=ax)
            ax.legend(loc="lower right", fontsize=8)

        # FIXME: This is a hack to make the x-axis labels show up correctly
        max_value = attn_weights.shape[1]
        percentages = ["0", "25", "50", "75", "100"]
        num_partitions = len(percentages)
        step = max_value / (num_partitions - 1)
        partitions = [i * step for i in range(num_partitions)]
        ax.set_xticks(partitions, ["0", "25", "50", "75", "100"])
        ax.set_xlim(0, max_value)

    if output_path is not None:
        file_extension = "png" if transparent else "jpg"
        plt.savefig(
            f"{output_path}/attn_weights_{mode}_decoder_layer.{file_extension}",
            dpi=150,
            bbox_inches="tight",
            transparent=transparent,
        )
    return fig, axes


def plot_decoder_attn_weights_bars(
    src_pose: Tensor,
    decoder_ca: dict[int, list[Tensor]],
    hp: HyperParameters,
    output_path: str,
    translation: list[str],
    layer: int = 0,
    norm_func: Callable = lambda t: (t - t.min()) / (t.max() - t.min()),
) -> FuncAnimation:
    """
    Displays keypoints as a pyplot visualization with an animated bar chart.
    Args:
        src_pose (Tensor): Tensor of shape (B, N, 2) where B is the batch size, N is the number of keypoints and 2 is the x and y coordinates.
        decoder_ca (dict[int, list[Tensor]]): Dictionary of tensors of shape (B, N, L) where B is the batch size, N is the number of words and L is the number of words in the source sentence.
        hp (HyperParameters): Hyperparameters object.
        output_path (str): Path to save the animation.
        translation (list[str]): List of words in the translation.
        layer (int): Layer of the decoder.
        batch_index (int): Index of the batch.
        norm_func (Callable): Function to normalize the attention weights.
    Returns:
        anim: Animation object.
    """

    def get_update(scatter, keypoints_all_frames, bars, attn_weights, hp):
        def update(frame):
            # Reshape keypoints to have each pair (x, y) in a row
            keypoints = keypoints_all_frames[frame, :].view(
                -1, (3 if hp["USE_3D"] else 2)
            )
            x = keypoints[:, 0]
            y = keypoints[:, 1]
            scatter.set_offsets(torch.stack((x, y), dim=-1))

            bar_heights = attn_weights[:, frame]
            for bar, height in zip(bars, bar_heights):
                bar.set_height(height)

            return scatter, bars

        return update

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()  # Invert y-axis so origin is at top-left

    translation_filtered = translation[1:]  # Remove BOS
    ax.set_title(" ".join(translation_filtered))

    scatter = ax.scatter([], [], s=10)

    ax_bar = fig.add_axes((0.65, 0.63, 0.25, 0.25))
    attn_weights = preprocess_attn_weights(decoder_ca[layer], norm_func)
    bars = ax_bar.bar(translation_filtered, attn_weights[:, 0], color="blue")
    ax_bar.set_ylim(0, attn_weights.max().item())
    ax_bar.set_xticks(range(len(translation_filtered)))
    ax_bar.set_xticklabels(translation_filtered, rotation=90)

    func_update = get_update(scatter, src_pose, bars, attn_weights, hp)

    num_frames = src_pose.shape[0]
    anim = FuncAnimation(fig, func_update, frames=num_frames, interval=50, blit=True)
    anim.save(
        f"{output_path}/sample_w_attn_weights.mp4",
        writer="ffmpeg",
    )

    return anim
