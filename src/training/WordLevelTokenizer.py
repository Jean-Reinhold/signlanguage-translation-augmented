from typing import Optional, Literal

import torch
import numpy as np


class WordLevelTokenizer:

    def __init__(self):
        self.token_to_idx: dict[str, int] = {
            "PAD": 0,
            "UNK": 1,
            "BOS": 2,
            "EOS": 3,
        }
        self.idx_to_token: dict[int, str] = {
            0: "PAD",
            1: "UNK",
            2: "BOS",
            3: "EOS",
        }
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.vocab_size = len(self.token_to_idx)

    def fit(self, texts: list[str]) -> None:
        for text in texts:
            for word in text.split():
                if word not in self.token_to_idx:
                    self.token_to_idx[word] = len(self.token_to_idx)
                    self.idx_to_token[len(self.idx_to_token)] = word
                    self.vocab_size += 1

    def tokenize(self, text: str) -> list[str]:
        return ["BOS"] + text.split() + ["EOS"]

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [
            (
                self.token_to_idx[token]
                if token in self.token_to_idx
                else self.unk_token_id
            )
            for token in tokens
        ]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.idx_to_token[id] for id in ids]

    def decode(self, ids: list[int], skip_special_tokens=False) -> str:
        if skip_special_tokens:
            ids = [
                id
                for id in ids
                if id not in [self.cls_token_id, self.sep_token_id, self.pad_token_id]
            ]
        return " ".join(self.convert_ids_to_tokens(ids))

    def encode(self, text: str, padding_len=0) -> list[int]:
        tokenized_text = self.tokenize(text)
        if padding_len > 0:
            tokenized_text = tokenized_text[:padding_len]
            tokenized_text += ["PAD"] * (padding_len - len(tokenized_text))
        return self.convert_tokens_to_ids(tokenized_text)

    def __call__(
        self,
        texts: list[str],
        padding: Literal["max_length", "longest", None] = "max_length",
        max_length: Optional[int] = None,
        return_tensors: Literal["pt", "np", None] = None,
    ) -> torch.Tensor | list[list[int]] | np.ndarray:
        padding_len = 0
        if padding == "max_length" and max_length is not None:
            padding_len = max_length
        elif padding == "longest":
            padding_len = max(map(len, [self.tokenize(text) for text in texts]))
        ids = [self.encode(text, padding_len) for text in texts]

        if return_tensors == "pt":
            return torch.tensor(ids)
        elif return_tensors == "np":
            return np.array(ids)
        else:
            return ids
