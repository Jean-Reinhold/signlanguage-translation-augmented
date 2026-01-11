import torch
from torch import Tensor


def generate_square_subsequent_mask(size: int, device: torch.device) -> Tensor:
    """
    Generates triangular (size, size) mask for the transformer model.
    """
    mask = (torch.triu(torch.ones((size, size))) == 1).transpose(0, 1).to(device)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_target_mask(
    tgt: Tensor, pad_idx: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """
    Create target mask and padding mask for the transformer model.
    Args:
        tgt: (N, T) where N is the batch size and T is the target sequence length
        pad_idx: padding index
        device: torch device
    Returns:
        tgt_mask: (T, T), so to evaluate the i-th token, we can only look at the first i tokens, for all i's
        tgt_padding_mask: (N, T), for masking pad tokens
    """
    tgt_seq_len = tgt.shape[1]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    tgt_padding_mask = tgt == pad_idx  # .type_as(tgt_mask)
    return tgt_mask, tgt_padding_mask


def create_src_mask(src: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
    """
    Create source padding mask for the transformer model.
    Args:
        src: (N, S, E) where N is the batch size, S is the source sequence length and E is the embedding size
        device: torch device
    Returns:
        src_mask: (N, S, S), by default, all frames can see all frames, so all values set to 0.
        src_padding_mask: (N, S), for masking padding frames. A frame is padded if all its keypoints are zero.
    """
    src_seq_len = src.shape[1]
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device)
    src_padding_mask = (src.sum(dim=-1) == 0).bool().to(device)
    return src_mask, src_padding_mask
