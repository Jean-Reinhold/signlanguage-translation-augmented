import math
import sys
import os

# Add parent directories to path for local imports
_src_path = os.path.dirname(os.path.dirname(__file__))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import torch
from typing import Optional
from torch import Tensor, nn
from torch.nn.functional import relu, softmax

from interp.InterpTransformer import InterpTransformer


class Conv1DEmbedder(nn.Module):
    """
    Apply 1D Convolutional layer to embed the keypoints to fit the input shape of the transformer.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(Conv1DEmbedder, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels, 128, 1)
        self.conv1d_4 = nn.Conv1d(128, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
                x: Tensor of shape (N, S, E)
        Returns:
                Tensor of shape (N, S, H), where the last dimension is the output of the convolutional layer to be used as input for the transformer.
        """
        x = x.permute(0, 2, 1)
        x = relu(self.conv1d_1(x))
        x = relu(self.conv1d_4(x))
        return x.permute(0, 2, 1)


class PositionalEncoding(nn.Module):
    """Code taken from https://pytorch.org/tutorials/beginner/translation_transformer.html"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply positional encoding to the input tensor.
        Args:
            x: Tensor of shape (N, S, E)
        Returns:
            Tensor of shape (N, S, E) where the positional encoding has been added to the input tensor.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """Code taken from https://pytorch.org/tutorials/beginner/translation_transformer.html"""

    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        """
        Applies token embedding to the target tensor.
        Args:
            tokens: Tensor of shape (N, T)
        Returns:
            Tensor of shape (N, T, H) after applying the embedding to the input tensor.
        """
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class KeypointsTransformer(nn.Module):
    """
    Transformer model for sign language translation. It uses a 1D convolutional layer to embed the keypoints and a transformer to translate the sequence.
    - S refers to the source sequence length (src_len).
    - T to the target sequence length (tgt_len).
    - E is the source feature amount (in_features).
    - N to the batch size.
    - H to the size of the dimension of the transformer embeddings (d_model).
    """

    def __init__(
        self,
        src_len: int,
        tgt_len: int,
        in_features: int,
        tgt_vocab_size: int,
        d_model: int = 64,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        interp: bool = False,
    ):
        """
        Args:
            src_max_len: max length of the source sequence
            tgt_max_len: max length of the target sequence
            in_features: number of features of the input (amount of keypoints * amount of coordinates)
            tgt_vocab_size: size of the target vocabulary
            d_model: number of dimensions of the encoding vectors (default=64). Must be even so the positional encoding works.
            num_encoder_layers: number of encoder layers (default=6)
            num_decoder_layers: number of decoder layers (default=6)
            dropout: dropout rate (default=0.1)
        """
        super().__init__()

        self.src_keyp_emb = Conv1DEmbedder(
            in_channels=in_features, out_channels=d_model
        )
        self.src_pe = PositionalEncoding(d_model=d_model, max_len=src_len)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.tgt_pe = PositionalEncoding(d_model=d_model, max_len=tgt_len)
        if not interp:
            self.transformer = nn.Transformer(
                d_model=d_model,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.transformer = InterpTransformer(
                d_model=d_model,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dropout=dropout,
                batch_first=True,
            )
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def embed_src(self, src: Tensor) -> Tensor:
        """
        Embed the source tensor.
        Args:
            src: Tensor of shape (N, S, E)
        Returns:
            Tensor of shape (N, S, H) representing the embedded source tensor
        """
        src_emb: Tensor = self.src_keyp_emb(src)

        assert (
            src_emb.shape[-1] == self.transformer.d_model
        ), f"Source embedding shape ({src_emb.shape[-1]}) should match {self.transformer.d_model} (H)"
        return src_emb

    def embed_tgt(
        self,
        tgt: Tensor,
    ) -> Tensor:
        """
        Embed the target tensor.
        Args:
            tgt: Tensor of shape (N, T)
        Returns:
            Tensor of shape (N, T, H) representing the embedded target tensor
        """
        tgt_emb = self.tgt_tok_emb(tgt)

        assert (
            tgt_emb.shape[-1] == self.transformer.d_model
        ), f"Target embedding shape ({tgt_emb.shape[-1]}) should match {self.transformer.d_model} (H)"
        return tgt_emb

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_mask: Tensor,
        tgt_padding_mask: Tensor,
    ) -> Tensor:
        """
        Forward pass of the model.
        Args:
            src: Tensor of shape (N, S, E)
            tgt: Tensor of shape (N, T)
            src_mask: Tensor of shape (N, S, S)
            src_padding_mask: Tensor of shape (N, S)
            tgt_mask: Tensor of shape (T, T)
            tgt_padding_mask: Tensor of shape (N, T)
        Returns:
            Tensor of shape (N, T, tgt_vocab_size) representing the output of the model
        """
        assert (
            src.dim() == 3
        ), f"Source tensor should have 3 dimensions, got {src.dim()}"
        assert (
            tgt.dim() == 2
        ), f"Target tensor should have 2 dimensions, got {tgt.dim()}"
        assert (
            tgt_mask.dim() == 2
        ), f"Target mask should have 2 dimensions, got {tgt_mask.dim()}"
        assert (
            tgt_padding_mask.dim() == 2
        ), f"Target padding mask should have 2 dimensions, got {tgt_padding_mask.dim()}"
        assert (
            len(
                set(
                    [
                        tgt.size(1),
                        tgt_mask.size(0),
                        tgt_mask.size(1),
                        tgt_padding_mask.size(1),
                    ]
                )
            )
            == 1
        ), f"Target dimensions mismatch: {tgt.size(1)=}, {tgt_mask.size()=}, {tgt_padding_mask.size()=}"

        src_emb = self.embed_src(src)
        tgt_emb = self.embed_tgt(tgt)
        src_emb = self.src_pe(src_emb)
        tgt_emb = self.tgt_pe(tgt_emb)
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.generator(outs)

    def encode(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src: Tensor of shape (N, S, E)
            src_mask: (S, S) mask
            src_padding_mask: (N, S) mask
        Returns:
            Tensor of shape (N, S, H) representing the output of the encoder
        """
        assert (
            src.dim() == 3
        ), f"Source tensor should have 3 dimensions, got {src.dim()}"

        src_emb = self.embed_src(src)
        src_emb = self.src_pe(src_emb)
        return self.transformer.encoder(src_emb, src_mask, src_padding_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_padding_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt: Integer tensor of shape (N, T)
            memory: Tensor of shape (N, S, H) resulting from the encoder
            memory_padding_mask: (N, S) mask. Typically should be the same as src_padding_mask
            tgt_mask: (T, T) mask
            tgt_padding_mask: (N, T) mask
        Returns:
            Tensor of shape (N, T, H) representing the output of the decoder
        """
        assert (
            tgt.dim() == 2
        ), f"Target tensor should have 2 dimensions, got {tgt.dim()}"
        tgt_emb = self.embed_tgt(tgt)
        tgt_emb = self.tgt_pe(tgt_emb)
        return self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            memory_key_padding_mask=memory_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

    def predict_proba(
        self,
        tgt: Tensor,
        src: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            tgt: Integer tensor of shape (N, T)
            src: Optional (N, S, E) shaped tensor
            src_padding_mask: Optional (N, S) shaped tensor
            memory: Optional (N, S, H) shaped tensor. If provided, src is ignored
            memory_padding_mask: Optional (N, S) shaped tensor. If provided, src_padding_mask is ignored
        Returns:
            Tuple of two tensors:
                - Tensor of shape (N, T, tgt_vocab_size) representing the output of the model as a probabilty distribution (after softmax)
                - Tensor of shape (N, S, H) representing the output of the encoder
        Raises:
            ValueError: if neither src nor memory is provided
        """
        assert (
            tgt.dim() == 2
        ), f"Target tensor should have 2 dimensions, got {tgt.dim()}"
        assert (
            src is None or src.dim() == 3
        ), f"Source tensor should have 3 dimensions, got {src.dim() if src is not None else None}"
        assert (
            memory is None or memory.dim() == 3
        ), f"Memory tensor should have 3 dimensions, got {memory.dim() if memory is not None else None}"

        if memory is not None:
            out = self.decode(tgt, memory, memory_padding_mask=memory_padding_mask)
        elif src is not None:
            memory = self.encode(src, src_padding_mask=src_padding_mask)
            out = self.decode(tgt, memory, memory_padding_mask=src_padding_mask)
        else:
            raise ValueError("Either src or memory must be provided")
        return (softmax(self.generator(out)[:, -1], dim=1), memory)
