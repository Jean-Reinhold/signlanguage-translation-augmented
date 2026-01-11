import sys
import os

# Add parent directories to path for local imports
_src_path = os.path.dirname(os.path.dirname(__file__))
_lib_path = os.path.join(os.path.dirname(_src_path), "lib")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)
if _lib_path not in sys.path:
    sys.path.insert(0, _lib_path)

import torch
import pandas as pd
from torch import Tensor
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy
from torchmetrics.functional.text import bleu_score, wer
import lightning as L

from hyperparameters import HyperParameters
from helpers import create_target_mask, create_src_mask
from .Translator import Translator
from .KeypointsTransformer import KeypointsTransformer
from .WordLevelTokenizer import WordLevelTokenizer
from posecraft.Pose import Pose


class LKeypointsTransformer(L.LightningModule):

    def __init__(
        self,
        hp: HyperParameters,
        device: torch.device,
        tokenizer: WordLevelTokenizer,
        interp: bool = False,
        class_weights: Tensor | None = None,
    ):
        super().__init__()

        # Model definition
        num_keypoints = Pose.get_components_mask(hp["LANDMARKS_USED"]).sum().item()
        in_features = int(num_keypoints * (3 if hp["USE_3D"] else 2))
        self.hp = hp
        self.model = KeypointsTransformer(
            src_len=hp["MAX_FRAMES"],
            tgt_len=hp["MAX_TOKENS"],
            in_features=in_features,
            tgt_vocab_size=tokenizer.vocab_size,
            d_model=hp["D_MODEL"],
            num_encoder_layers=hp["NUM_ENCODER_LAYERS"],
            num_decoder_layers=hp["NUM_DECODER_LAYERS"],
            dropout=hp["DROPOUT"],
            interp=interp,
        )

        # Other configurations
        self.loss_fn = cross_entropy
        self.running_device = device
        self.tokenizer = tokenizer
        self.translator = Translator(device, hp["MAX_TOKENS"])
        self.accuracy = Accuracy(
            task="multiclass",
            num_classes=self.tokenizer.vocab_size,
            ignore_index=tokenizer.pad_token_id,
        )
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.running_device)
        self.example_input_array = self.get_sample_input(in_features)

        # List initialization for translation results during test
        self.ys_step: list[str] = []
        self.beam_translations_step: list[str] = []
        self.greedy_translations_step: list[str] = []
        self.translation_results_df: pd.DataFrame | None = None

        self.save_hyperparameters()

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_mask: Tensor,
        tgt_padding_mask: Tensor,
    ):
        return self.model(
            src, tgt, src_mask, src_padding_mask, tgt_mask, tgt_padding_mask
        )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hp["LR"])
        return optimizer

    def run_on_batch(self, batch):
        src, tgt = batch
        # tgt_input and tgt_ouptut are displaced by one position, so tgt_input[i] is the input to the model and tgt_output[i] is the expected output
        tgt_input: Tensor = tgt[:, :-1]
        tgt_mask, tgt_padding_mask = create_target_mask(
            tgt_input, self.tokenizer.pad_token_id, self.running_device
        )
        src_mask, src_padding_mask = create_src_mask(src, self.running_device)
        logits: Tensor = self.model(
            src, tgt_input, src_mask, src_padding_mask, tgt_mask, tgt_padding_mask
        )
        tgt_output: Tensor = tgt[:, 1:]
        loss = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            tgt_output.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
            weight=self.class_weights,
        )
        accuracy = self.accuracy(
            logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)
        )
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.run_on_batch(batch)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_accuracy", accuracy, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.run_on_batch(batch)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_accuracy", accuracy, on_epoch=True, batch_size=len(batch))
        return loss

    def test_step(self, batch, batch_idx):
        self.model.eval()
        loss, accuracy = self.run_on_batch(batch)
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        ys, preds_greedy, preds_beam = self.get_translations(batch, batch_idx)
        self.ys_step.extend(ys)
        self.greedy_translations_step.extend(preds_greedy)
        self.beam_translations_step.extend(preds_beam)
        return loss, accuracy

    def on_test_epoch_end(self):
        translation_results = [
            (y, trans_greedy, trans_beam, wer.word_error_rate(trans_greedy, y).item())
            + tuple(bleu_score(trans_greedy, [y], n_gram=n).item() for n in range(1, 5))
            + tuple(bleu_score(trans_beam, [y], n_gram=n).item() for n in range(1, 5))
            for y, trans_greedy, trans_beam in zip(
                self.ys_step, self.greedy_translations_step, self.beam_translations_step
            )
        ]
        self.ys_step = []
        self.greedy_translations_step = []
        self.beam_translations_step = []
        translation_results_df = pd.DataFrame(
            translation_results,
            columns=[
                "y",
                "trans_greedy",
                "trans_beam",
                "wer",
                "bleu_1_greedy",
                "bleu_2_greedy",
                "bleu_3_greedy",
                "bleu_4_greedy",
                "bleu_1_beam",
                "bleu_2_beam",
                "bleu_3_beam",
                "bleu_4_beam",
            ],
        )
        self.logger.log_table(key="translation-results", columns=list(translation_results_df.columns), data=translation_results)  # type: ignore
        self.log("wer", translation_results_df["wer"].mean())
        self.log("bleu_1_greedy", translation_results_df["bleu_1_greedy"].mean())
        self.log("bleu_2_greedy", translation_results_df["bleu_2_greedy"].mean())
        self.log("bleu_3_greedy", translation_results_df["bleu_3_greedy"].mean())
        self.log("bleu_4_greedy", translation_results_df["bleu_4_greedy"].mean())
        self.log("bleu_1_beam", translation_results_df["bleu_1_beam"].mean())
        self.log("bleu_2_beam", translation_results_df["bleu_2_beam"].mean())
        self.log("bleu_3_beam", translation_results_df["bleu_3_beam"].mean())
        self.log("bleu_4_beam", translation_results_df["bleu_4_beam"].mean())
        self.translation_results_df = translation_results_df

    def get_translations(
        self, batch: Tensor, batch_idx: int
    ) -> tuple[list[str], list[str], list[str]]:
        src, tgt = batch
        preds_greedy = self.translator.translate(
            src, self.model, "greedy", self.tokenizer
        )
        preds_beam = self.translator.translate(
            src, self.model, "beam", self.tokenizer, k=32
        )
        ys: list[str] = []
        for i in range(len(src)):
            ys.append(
                self.tokenizer.decode(
                    [int(x) for x in tgt[i].tolist()],
                    skip_special_tokens=True,
                )
            )
        return ys, preds_greedy, preds_beam

    def get_sample_input(
        self, input_features: int, batch_size: int = 1
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        sample_src = torch.randn(batch_size, self.hp["MAX_FRAMES"], input_features)
        sample_tgt = torch.randint(
            0, self.tokenizer.vocab_size, (batch_size, self.hp["MAX_TOKENS"])
        )
        sample_src_mask, sample_src_padding_mask = create_src_mask(
            sample_src, self.device
        )
        sample_tgt_mask, sample_tgt_padding_mask = create_target_mask(
            sample_tgt, self.tokenizer.pad_token_id, self.device
        )
        return (
            sample_src,
            sample_tgt,
            sample_src_mask,
            sample_src_padding_mask,
            sample_tgt_mask,
            sample_tgt_padding_mask,
        )
