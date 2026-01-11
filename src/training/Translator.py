import sys
import os
from typing import Literal, Optional

# Add parent directories to path for local imports
_src_path = os.path.dirname(os.path.dirname(__file__))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import torch
from torch import Tensor

from .WordLevelTokenizer import WordLevelTokenizer
from .KeypointsTransformer import KeypointsTransformer
from helpers import create_src_mask


class Translator:
    def __init__(
        self,
        device: torch.device,
        max_tokens: int,
    ):
        self.device = device
        self.max_tokens = max_tokens

    def translate(
        self,
        src: Tensor,
        model: KeypointsTransformer,
        method: Literal["greedy", "beam"],
        tokenizer: WordLevelTokenizer,
        k: Optional[int] = 32,
    ) -> list[str]:
        """
        Translate a batch of sequences using the specified method
        Args:
            src: (N, S) integer tensor
            model: model to use for translation
            method: decoding method, either "greedy" or "beam"
            tokenizer: tokenizer to decode the output sequences
            max_tokens: maximum number of tokens to generate
            device: device to use for the model
            k: beam size
        Returns:
            List of strings containing the translated sequences
        """
        model.eval()
        with torch.no_grad():
            if method == "greedy":
                out = self.greedy_decode(
                    src, model, tokenizer.cls_token_id, tokenizer.sep_token_id
                )
            elif method == "beam":
                if k is None:
                    raise ValueError(
                        "Beam size must be specified when using beam search."
                    )
                out = self.beam_decode(
                    src, model, k, tokenizer.cls_token_id, tokenizer.sep_token_id
                )
            else:
                raise ValueError("Invalid method. Choose between 'greedy' and 'beam'.")
        return [
            tokenizer.decode(
                [int(x) for x in out[i].tolist()], skip_special_tokens=True
            )
            for i in range(len(src))
        ]

    def greedy_decode(
        self,
        src: Tensor,
        model: KeypointsTransformer,
        bos: int,
        eos: int,
    ) -> Tensor:
        """
        Greedy decoding algorithm for batched inputs
        Args:
            src: (N, S) integer tensor
            model: model to use for translation
            bos: beginning of sentence token
            eos: end of sentence token
        Returns:
            Tensor of shape (N, T) containing the output of the model until the end token or the maximum number of tokens is reached
        """
        src_mask, src_padding_mask = create_src_mask(src, self.device)
        ys = torch.full((src.size(0), 1), bos).to(self.device)
        prob, memory = model.predict_proba(
            ys, src=src, src_padding_mask=src_padding_mask
        )
        for _ in range(self.max_tokens - 1):
            _, next_word = torch.max(prob, dim=1)
            # if the last word is the end token, keep it and do not predict the next word
            next_word = torch.where(
                ys[:, -1] == eos,
                torch.full(next_word.shape, eos).to(self.device),
                next_word,
            )
            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)
            if (next_word == eos).all():
                break
            prob, memory = model.predict_proba(
                ys, memory=memory, memory_padding_mask=src_padding_mask
            )
        return ys

    def single_beam_decode(
        self, src: Tensor, model: KeypointsTransformer, k: int, bos: int, eos: int
    ) -> Tensor:
        """
        Beam search decoding algorithm for a single sample
        Args:
            src: (S) integer tensor
            model: model to use for translation
            k: beam size
        Returns:
            Tensor of shape (T) containing the output of the model until the end token or the maximum number of tokens is reached
        """
        # first dimension corresponding to the batch will be used to predict over the k posible beams
        src = src.repeat(k, 1, 1)
        # ys is a Tensor of shape (k, T), initialized with the start token and beams_probs is a Tensor of shape (k) with the probabilities of each beam
        ys = torch.full((k, 1), bos).to(self.device)
        beams_probs = torch.ones(k).to(self.device)
        src_mask, src_padding_mask = create_src_mask(src, self.device)

        next_word_probs, memory = model.predict_proba(
            ys, src=src, src_padding_mask=src_padding_mask
        )

        for _ in range(self.max_tokens - 1):
            # repeat the possible ys and the probabilities k times to later generate k combinations for each beam
            # repeated_ys is a Tensor of shape (k*k, T), repeated_beams_probs is a Tensor of shape (k*k)
            repeated_ys = ys.repeat_interleave(k, dim=0)
            repeated_beams_probs = beams_probs.repeat_interleave(k, dim=0).view(-1)

            # next_words_probs and next_words are Tensors of shape (k,k), the first is per each beam, the second the top k words
            # they are shaped as (k*k) to combine the possible ys with the top k words and their probabilities
            next_words_probs, next_words = torch.topk(next_word_probs, k=k)
            next_words = next_words.view(-1)
            next_words_probs = next_words_probs.view(-1)

            # if any possible y is the end token, change the following token by another end token and keep the original probabilties
            next_words_probs = torch.where(
                repeated_ys[:, -1] == eos,
                torch.ones_like(next_words_probs),
                next_words_probs,
            )
            next_words = torch.where(
                repeated_ys[:, -1] == eos,
                torch.full_like(next_words, eos),
                next_words,
            )
            # next_possible_ys is a Tensor of shape (k*k, T+1) with the next possible sequences and next_beam_probs is a Tensor of shape (k*k) with the probabilities for each sequence
            next_possible_ys = torch.cat((repeated_ys, next_words.unsqueeze(1)), dim=1)
            next_beam_probs = next_words_probs * repeated_beams_probs

            # remove repeated sequences from the possible_ys and probs
            l = [str(x) for x in next_possible_ys.tolist()]  # refactor this
            unique_indices = [l.index(x) for x in set(l)]
            next_possible_ys = next_possible_ys[unique_indices]
            next_beam_probs = next_beam_probs[unique_indices]

            # sort the sequences by their probabilities and keep the top k
            sorted_indices = torch.argsort(next_beam_probs, descending=True)
            beams_probs = torch.index_select(next_beam_probs, 0, sorted_indices)[:k]
            ys = torch.index_select(next_possible_ys, 0, sorted_indices)[:k]

            if (ys[:, -1] == eos).all():
                break

            next_word_probs, memory = model.predict_proba(
                ys, memory=memory, memory_padding_mask=src_padding_mask
            )
        return ys[0].squeeze()

    def beam_decode(
        self, src: Tensor, model: KeypointsTransformer, k: int, bos: int, eos: int
    ) -> Tensor:
        """
        Beam search decoding algorithm for batched inputs
        Args:
            src: (N, S) integer tensor
            model: model to use for translation
            k: beam size
        Returns:
            Tensor of shape (N, T) containing the output of the model until the end token or the maximum number of tokens is reached
        """
        # TODO: rewrite this function to do the beam search matrix-wise
        preds_beam = []
        for i in range(len(src)):
            src_0 = src[i]
            preds_beam.append(self.single_beam_decode(src_0, model, k, bos, eos))
        max_length = max(tensor.size(0) for tensor in preds_beam)
        padded_tensors = [
            torch.nn.functional.pad(tensor, (0, max_length - tensor.size(0)))
            for tensor in preds_beam
        ]
        return torch.stack(padded_tensors)
