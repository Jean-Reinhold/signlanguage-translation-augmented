#!/usr/bin/env python
import torch

torch.backends.cudnn.deterministic = True

import argparse
import numpy as np
import os
import shutil
import time
import queue

from main.model import build_model
from main.batch import Batch
from main.helpers import (
    log_data_info,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger,
    set_seed,
    symlink_update,
)
from main.model import SignModel
from main.prediction import validate_on_data
from main.dataset import SignTranslationDataset
from main.loss import XentLoss
from main.data import load_data, make_data_iter
from main.dataset import iter_dataset_file
from torchtext import data as ttdata
from main.builders import build_optimizer, build_scheduler, build_gradient_clipper
from main.prediction import test
from main.metrics import wer_single
from main.vocabulary import SIL_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Dataset
from typing import List, Dict

# pylint: disable=too-many-instance-attributes
class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""

    def __init__(self, model: SignModel, config: dict) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        """
        train_config = config["training"]

        # files for logging and storing
        self.model_dir = make_model_dir(
            train_config["model_dir"], overwrite=train_config.get("overwrite", False)
        )
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir + "/tensorboard/")

        # input
        self.feature_size = (
            sum(config["data"]["feature_size"])
            if isinstance(config["data"]["feature_size"], list)
            else config["data"]["feature_size"]
        )
        self.dataset_version = config["data"].get("version", "phoenix_2014_trans")

        # model
        self.model = model
        self.txt_pad_index = self.model.txt_pad_index
        self.txt_bos_index = self.model.txt_bos_index
        self._log_parameters_list()
        # Check if we are doing only recognition or only translation or both
        self.do_recognition = (
            config["training"].get("recognition_loss_weight", 1.0) > 0.0
        )
        self.do_translation = (
            config["training"].get("translation_loss_weight", 1.0) > 0.0
        )

        # Get Recognition and Translation specific parameters
        if self.do_recognition:
            self._get_recognition_params(train_config=train_config)
        if self.do_translation:
            self._get_translation_params(train_config=train_config)

        # optimization
        self.last_best_lr = train_config.get("learning_rate", -1)
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)

        params = model.parameters()
        self.optimizer = build_optimizer(
            config=train_config, parameters=params
        )
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 100)
        self.num_valid_log = train_config.get("num_valid_log", 5)
        self.ckpt_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 5))
        self.eval_metric = train_config.get("eval_metric", "bleu")
        if self.eval_metric not in ["bleu", "chrf", "wer", "rouge"]:
            raise ValueError(
                "Invalid setting for 'eval_metric': {}".format(self.eval_metric)
            )
        self.early_stopping_metric = train_config.get(
            "early_stopping_metric", "eval_metric"
        )

        # if we schedule after BLEU/chrf, we want to maximize it, else minimize
        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric
        if self.early_stopping_metric in [
            "ppl",
            "translation_loss",
            "recognition_loss",
        ]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            if self.eval_metric in ["bleu", "chrf", "rouge"]:
                assert self.do_translation
                self.minimize_metric = False
            else:  # eval metric that has to get minimized (not yet implemented)
                self.minimize_metric = True
        else:
            raise ValueError(
                "Invalid setting for 'early_stopping_metric': {}".format(
                    self.early_stopping_metric
                )
            )

        # data_augmentation parameters
        self.frame_subsampling_ratio = config["data"].get(
            "frame_subsampling_ratio", None
        )
        self.random_frame_subsampling = config["data"].get(
            "random_frame_subsampling", None
        )
        self.random_frame_masking_ratio = config["data"].get(
            "random_frame_masking_ratio", None
        )

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"],
        )

        # data & batch handling
        self.level = config["data"]["level"]
        if self.level not in ["word", "bpe", "char"]:
            raise ValueError("Invalid segmentation level': {}".format(self.level))
        # Keep max sentence length for filtering when streaming shards
        self.max_sent_length = config["data"].get("max_sent_length", 400)
        # Chunk size when streaming training data
        self.stream_chunk_size = config["data"].get("stream_chunk_size", 5000)
        
        # Temperature-based sampling for multilingual training (v5 feature)
        # T=1: proportional to dataset size (large datasets dominate)
        # T=5: balanced (standard for multilingual, e.g., mT5, mBART)
        # T=∞: uniform sampling
        self.sampling_temperature = config["data"].get("sampling_temperature", 5.0)
        self.logger.info("Multilingual sampling temperature: %.1f", self.sampling_temperature)

        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_type = train_config.get("batch_type", "sentence")
        self.eval_batch_size = train_config.get("eval_batch_size", self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type", self.batch_type)

        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
            if self.do_translation:
                self.translation_loss_function.cuda()
            if self.do_recognition:
                self.recognition_loss_function.cuda()

        # initialize training statistics
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_txt_tokens = 0
        self.total_gls_tokens = 0
        self.best_ckpt_iteration = 0
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        self.best_all_ckpt_scores = {}
        # comparison function for scores
        self.is_best = (
            lambda score: score < self.best_ckpt_score
            if self.minimize_metric
            else score > self.best_ckpt_score
        )

        # model parameters
        if "load_model" in train_config.keys():
            model_load_path = train_config["load_model"]
            self.logger.info("Loading model from %s", model_load_path)
            reset_best_ckpt = train_config.get("reset_best_ckpt", False)
            reset_scheduler = train_config.get("reset_scheduler", False)
            reset_optimizer = train_config.get("reset_optimizer", False)
            self.init_from_checkpoint(
                model_load_path,
                reset_best_ckpt=reset_best_ckpt,
                reset_scheduler=reset_scheduler,
                reset_optimizer=reset_optimizer,
            )

    def _get_recognition_params(self, train_config) -> None:
        # NOTE (Cihan): The blank label is the silence index in the gloss vocabulary.
        #   There is an assertion in the GlossVocabulary class's __init__.
        #   This is necessary to do TensorFlow decoding, as it is hardcoded
        #   Currently it is hardcoded as 0.
        self.gls_silence_token = self.model.gls_vocab.stoi[SIL_TOKEN]
        assert self.gls_silence_token == 0

        self.recognition_loss_function = torch.nn.CTCLoss(
            blank=self.gls_silence_token, zero_infinity=True
        )
        self.recognition_loss_weight = train_config.get("recognition_loss_weight", 1.0)
        self.eval_recognition_beam_size = train_config.get(
            "eval_recognition_beam_size", 1
        )

    def _get_translation_params(self, train_config) -> None:
        self.label_smoothing = train_config.get("label_smoothing", 0.0)
        self.translation_loss_function = XentLoss(
            pad_index=self.txt_pad_index, smoothing=self.label_smoothing
        )
        self.translation_normalization_mode = train_config.get(
            "translation_normalization", "batch"
        )
        if self.translation_normalization_mode not in ["batch", "tokens"]:
            raise ValueError(
                "Invalid normalization {}.".format(self.translation_normalization_mode)
            )
        self.translation_loss_weight = train_config.get("translation_loss_weight", 1.0)
        self.eval_translation_beam_size = train_config.get(
            "eval_translation_beam_size", 1
        )
        self.eval_translation_beam_alpha = train_config.get(
            "eval_translation_beam_alpha", -1
        )
        self.translation_max_output_length = train_config.get(
            "translation_max_output_length", None
        )

    def _save_checkpoint(self) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint.

        The training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far,
        and optimizer and scheduler states.

        """
        model_path = "{}/{}.ckpt".format(self.model_dir, self.steps)
        state = {
            "steps": self.steps,
            "total_txt_tokens": self.total_txt_tokens if self.do_translation else 0,
            "total_gls_tokens": self.total_gls_tokens if self.do_recognition else 0,
            "best_ckpt_score": self.best_ckpt_score,
            "best_all_ckpt_scores": self.best_all_ckpt_scores,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
        }
        torch.save(state, model_path)
        if self.ckpt_queue.full():
            to_delete = self.ckpt_queue.get()  # delete oldest ckpt
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                self.logger.warning(
                    "Wanted to delete old checkpoint %s but " "file does not exist.",
                    to_delete,
                )

        self.ckpt_queue.put(model_path)

        # create/modify symbolic link for best checkpoint
        symlink_update(
            "{}.ckpt".format(self.steps), "{}/best.ckpt".format(self.model_dir)
        )

    def init_from_checkpoint(
        self,
        path: str,
        reset_best_ckpt: bool = False,
        reset_scheduler: bool = False,
        reset_optimizer: bool = False,
    ) -> None:
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also
        scheduler and optimizer states, see `self._save_checkpoint`.

        :param path: path to checkpoint
        :param reset_best_ckpt: reset tracking of the best checkpoint,
                                use for domain adaptation with a new dev
                                set or when using a new metric for fine-tuning.
        :param reset_scheduler: reset the learning rate scheduler, and do not
                                use the one stored in the checkpoint.
        :param reset_optimizer: reset the optimizer, and do not use the one
                                stored in the checkpoint.
        """
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            self.logger.info("Reset optimizer.")

        if not reset_scheduler:
            if (
                model_checkpoint["scheduler_state"] is not None
                and self.scheduler is not None
            ):
                self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])
        else:
            self.logger.info("Reset scheduler.")

        # restore counts
        self.steps = model_checkpoint["steps"]
        self.total_txt_tokens = model_checkpoint["total_txt_tokens"]
        self.total_gls_tokens = model_checkpoint["total_gls_tokens"]

        if not reset_best_ckpt:
            self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.best_all_ckpt_scores = model_checkpoint["best_all_ckpt_scores"]
            self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]
        else:
            self.logger.info("Reset tracking of the best checkpoint.")

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda()
        
        # Per-language validation (v4 feature)
        self.dev_per_language = {}  # Dict[str, Dataset]
    
    def set_per_language_validation(self, dev_per_language: dict) -> None:
        """
        Set per-language validation datasets for multilingual training.
        
        :param dev_per_language: Dict mapping language name to validation dataset
        """
        self.dev_per_language = dev_per_language
        self.logger.info("Per-language validation enabled for %d languages: %s",
                        len(dev_per_language), list(dev_per_language.keys()))

    def train_and_validate(self,  train_data: Dataset, valid_data: Dataset) -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        # Support streaming over training shards to reduce peak memory
        def _epoch_batches():
            if isinstance(train_data, dict):
                paths = train_data["paths"]
                fields = train_data["fields"]
                paths = paths if isinstance(paths, list) else [paths]
                named_fields = [
                    ("sequence", fields[0]),
                    ("signer", fields[1]),
                    ("sgn", fields[2]),
                    ("gls", fields[3]),
                    ("txt", fields[4]),
                ]
                
                # =========================================================
                # TRUE INTERLEAVED MULTILINGUAL BATCHING (v6)
                # =========================================================
                # Each batch contains samples from ALL languages to prevent
                # catastrophic forgetting. Uses temperature-weighted sampling
                # to balance high and low-resource languages.
                # =========================================================
                
                import random
                import os
                
                # Group paths by dataset/language
                dataset_paths = {}
                for p in paths:
                    basename = os.path.basename(p)
                    # Extract dataset name (handle .part* files)
                    if ".pami0" in basename:
                        dataset_name = basename.split(".pami0")[0]
                    else:
                        dataset_name = basename.split(".")[0]
                    if dataset_name not in dataset_paths:
                        dataset_paths[dataset_name] = []
                    dataset_paths[dataset_name].append(p)
                
                n_datasets = len(dataset_paths)
                self.logger.info("[train] Interleaved batching across %d datasets: %s", 
                               n_datasets, list(dataset_paths.keys()))
                
                # =========================================================
                # PHASE 1: Pre-fill buffers from all datasets
                # =========================================================
                # We maintain a buffer for each dataset and refill as needed.
                # Buffer size is chosen to allow good mixing while managing memory.
                # =========================================================
                
                BUFFER_SIZE_PER_DATASET = 1000  # Samples to keep in memory per dataset
                
                # Create sample generators for each dataset
                def _create_sample_generator(dataset_name):
                    """Generator that yields filtered samples from a dataset."""
                    for p in dataset_paths[dataset_name]:
                        self.logger.info("[train] Loading %s from %s", dataset_name, p)
                        for s in iter_dataset_file(p):
                            txt = s["text"].strip()
                            txt_len = len(list(txt)) if self.level == "char" else len(txt.split())
                            if len(s["sign"]) <= self.max_sent_length and txt_len <= self.max_sent_length:
                                yield s
                
                # Initialize generators and buffers
                generators = {name: _create_sample_generator(name) for name in dataset_paths}
                buffers = {name: [] for name in dataset_paths}
                active_datasets = set(dataset_paths.keys())
                samples_yielded = {name: 0 for name in dataset_paths}
                
                def _refill_buffer(name, target_size=BUFFER_SIZE_PER_DATASET):
                    """Refill a dataset's buffer up to target_size."""
                    if name not in active_datasets:
                        return False
                    try:
                        while len(buffers[name]) < target_size:
                            sample = next(generators[name])
                            buffers[name].append(sample)
                        return True
                    except StopIteration:
                        if not buffers[name]:
                            active_datasets.discard(name)
                            self.logger.info("[train] Dataset %s exhausted (%d samples)", 
                                           name, samples_yielded[name])
                        return len(buffers[name]) > 0
                
                # Initial buffer fill
                for name in list(active_datasets):
                    _refill_buffer(name)
                    self.logger.info("[train] Initial buffer for %s: %d samples", 
                                   name, len(buffers[name]))
                
                # Calculate temperature-weighted sampling probabilities
                T = getattr(self, 'sampling_temperature', 5.0)
                
                def _calc_sampling_weights():
                    """Calculate temperature-scaled sampling weights."""
                    if not active_datasets:
                        return {}
                    sizes = {n: max(len(buffers[n]), 1) for n in active_datasets}
                    # p(i) ∝ n_i^(1/T) - higher T = more uniform, lower T = more proportional
                    weights = {n: sizes[n] ** (1.0 / T) for n in active_datasets}
                    total = sum(weights.values())
                    return {n: w / total for n, w in weights.items()}
                
                # Log initial probabilities
                weights = _calc_sampling_weights()
                self.logger.info("[train] Temperature=%.1f, sampling weights: %s", 
                               T, {n: f"{w:.1%}" for n, w in weights.items()})
                
                # =========================================================
                # PHASE 2: Generate interleaved batches
                # =========================================================
                # For each batch, sample from ALL active datasets proportionally.
                # This ensures every gradient update sees all languages.
                # =========================================================
                
                batch_count = 0
                samples_per_batch = self.batch_size
                
                while active_datasets:
                    # Build a mixed batch from all active datasets
                    mixed_batch = []
                    weights = _calc_sampling_weights()
                    
                    if not weights:
                        break
                    
                    # Calculate how many samples to take from each dataset
                    # Ensure at least 1 sample from each active dataset
                    n_active = len(active_datasets)
                    min_per_dataset = max(1, samples_per_batch // (n_active * 2))
                    remaining = samples_per_batch
                    
                    samples_to_take = {}
                    for name in active_datasets:
                        # Take proportional amount based on weights
                        n_samples = max(min_per_dataset, int(samples_per_batch * weights[name]))
                        # But don't exceed what's available
                        n_samples = min(n_samples, len(buffers[name]))
                        samples_to_take[name] = n_samples
                    
                    # Adjust to hit exact batch size
                    total_planned = sum(samples_to_take.values())
                    if total_planned < samples_per_batch:
                        # Add more from largest buffers
                        deficit = samples_per_batch - total_planned
                        sorted_by_buffer = sorted(active_datasets, 
                                                 key=lambda n: len(buffers[n]), reverse=True)
                        for name in sorted_by_buffer:
                            can_add = len(buffers[name]) - samples_to_take[name]
                            add = min(can_add, deficit)
                            samples_to_take[name] += add
                            deficit -= add
                            if deficit <= 0:
                                break
                    
                    # Extract samples from each buffer
                    for name, n_samples in samples_to_take.items():
                        if n_samples > 0 and buffers[name]:
                            # Random sample from buffer
                            random.shuffle(buffers[name])
                            taken = buffers[name][:n_samples]
                            buffers[name] = buffers[name][n_samples:]
                            mixed_batch.extend(taken)
                            samples_yielded[name] += n_samples
                    
                    # Skip if batch is too small
                    if len(mixed_batch) < self.batch_size // 2:
                        # Try to refill and continue
                        any_refilled = False
                        for name in list(active_datasets):
                            if _refill_buffer(name):
                                any_refilled = True
                        if not any_refilled and not mixed_batch:
                            break
                        if not mixed_batch:
                            continue
                    
                    # Shuffle the mixed batch to interleave languages
                    random.shuffle(mixed_batch)
                    
                    # Convert to torchtext examples
                    examples = [
                        ttdata.Example.fromlist(
                            [
                                smp["name"],
                                smp["signer"],
                                smp["sign"] + 1e-8,
                                smp["gloss"].strip(),
                                smp["text"].strip(),
                            ],
                            named_fields,
                        )
                        for smp in mixed_batch
                    ]
                    
                    # Create dataset and iterator
                    dataset_chunk = ttdata.Dataset(examples, named_fields)
                    data_iter = make_data_iter(
                        dataset_chunk,
                        batch_size=self.batch_size,
                        batch_type=self.batch_type,
                        train=True,
                        shuffle=False,  # Already shuffled
                    )
                    
                    for b in iter(data_iter):
                        yield b
                        batch_count += 1
                    
                    del dataset_chunk
                    del mixed_batch
                    
                    # Refill depleted buffers
                    for name in list(active_datasets):
                        if len(buffers[name]) < BUFFER_SIZE_PER_DATASET // 2:
                            _refill_buffer(name)
                    
                    # Log progress periodically
                    if batch_count % 100 == 0:
                        active_info = {n: len(buffers[n]) for n in active_datasets}
                        self.logger.info("[train] Batch %d, active buffers: %s", 
                                       batch_count, active_info)
                
                # Final stats
                self.logger.info("[train] Epoch complete: %d batches", batch_count)
                self.logger.info("[train] Samples per dataset: %s", samples_yielded)
                
            else:
                data_iter = make_data_iter(
                    train_data,
                    batch_size=self.batch_size,
                    batch_type=self.batch_type,
                    train=True,
                    shuffle=self.shuffle,
                )
                for b in iter(data_iter):
                    yield b
        epoch_no = None
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()
            start = time.time()
            total_valid_duration = 0
            count = self.batch_multiplier - 1

            if self.do_recognition:
                processed_gls_tokens = self.total_gls_tokens
                epoch_recognition_loss = 0
            if self.do_translation:
                processed_txt_tokens = self.total_txt_tokens
                epoch_translation_loss = 0

            for batch in _epoch_batches():
                # reactivate training
                # create a Batch object from torchtext batch
                batch = Batch(
                    is_train=True,
                    torch_batch=batch,
                    txt_pad_index=self.txt_pad_index,
                    sgn_dim=self.feature_size,
                    use_cuda=self.use_cuda,
                    frame_subsampling_ratio=self.frame_subsampling_ratio,
                    random_frame_subsampling=self.random_frame_subsampling,
                    random_frame_masking_ratio=self.random_frame_masking_ratio,
                )

                # only update every batch_multiplier batches
                # see https://medium.com/@davidlmorton/
                # increasing-mini-batch-size-without-increasing-
                # memory-6794e10db672
                update = count == 0

                recognition_loss, translation_loss = self._train_batch(
                    batch, update=update
                )

                if self.do_recognition:
                    self.tb_writer.add_scalar(
                        "train/train_recognition_loss", recognition_loss, self.steps
                    )
                    epoch_recognition_loss += recognition_loss.detach().cpu().numpy()

                if self.do_translation:
                    self.tb_writer.add_scalar(
                        "train/train_translation_loss", translation_loss, self.steps
                    )
                    epoch_translation_loss += translation_loss.detach().cpu().numpy()

                count = self.batch_multiplier if update else count
                count -= 1

                if (
                    self.scheduler is not None
                    and self.scheduler_step_at == "step"
                    and update
                ):
                    self.scheduler.step()

                # log learning progress
                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration

                    log_out = "[Epoch: {:03d} Step: {:08d}] ".format(
                        epoch_no + 1, self.steps,
                    )

                    if self.do_recognition:
                        elapsed_gls_tokens = (
                            self.total_gls_tokens - processed_gls_tokens
                        )
                        processed_gls_tokens = self.total_gls_tokens
                        log_out += "Batch Recognition Loss: {:10.6f} => ".format(
                            recognition_loss
                        )
                        log_out += "Gls Tokens per Sec: {:8.0f} || ".format(
                            elapsed_gls_tokens / elapsed
                        )
                    if self.do_translation:
                        elapsed_txt_tokens = (
                            self.total_txt_tokens - processed_txt_tokens
                        )
                        processed_txt_tokens = self.total_txt_tokens
                        log_out += "Batch Translation Loss: {:10.6f} => ".format(
                            translation_loss
                        )
                        log_out += "Txt Tokens per Sec: {:8.0f} || ".format(
                            elapsed_txt_tokens / elapsed
                        )
                    log_out += "Lr: {:.6f}".format(self.optimizer.param_groups[0]["lr"])
                    self.logger.info(log_out)
                    start = time.time()
                    total_valid_duration = 0

                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and update:
                    valid_start_time = time.time()
                    # TODO (Cihan): There must be a better way of passing
                    #   these recognition only and translation only parameters!
                    #   Maybe have a NamedTuple with optional fields?
                    #   Hmm... Future Cihan's problem.
                    val_res = validate_on_data(
                        model=self.model,
                        data=valid_data,
                        batch_size=self.eval_batch_size,
                        use_cuda=self.use_cuda,
                        batch_type=self.eval_batch_type,
                        dataset_version=self.dataset_version,
                        sgn_dim=self.feature_size,
                        txt_pad_index=self.txt_pad_index,
                        # Recognition Parameters
                        do_recognition=self.do_recognition,
                        recognition_loss_function=self.recognition_loss_function
                        if self.do_recognition
                        else None,
                        recognition_loss_weight=self.recognition_loss_weight
                        if self.do_recognition
                        else None,
                        recognition_beam_size=self.eval_recognition_beam_size
                        if self.do_recognition
                        else None,
                        # Translation Parameters
                        do_translation=self.do_translation,
                        translation_loss_function=self.translation_loss_function
                        if self.do_translation
                        else None,
                        translation_max_output_length=self.translation_max_output_length
                        if self.do_translation
                        else None,
                        level=self.level if self.do_translation else None,
                        translation_loss_weight=self.translation_loss_weight
                        if self.do_translation
                        else None,
                        translation_beam_size=self.eval_translation_beam_size
                        if self.do_translation
                        else None,
                        translation_beam_alpha=self.eval_translation_beam_alpha
                        if self.do_translation
                        else None,
                        frame_subsampling_ratio=self.frame_subsampling_ratio,
                    )
                    self.model.train()
                    self.tb_writer.add_scalar(
                        "learning_rate",
                        self.scheduler.optimizer.param_groups[0]["lr"],
                        self.steps,
                    )
                    if self.do_recognition:
                        # Log Losses and ppl
                        self.tb_writer.add_scalar(
                            "valid/valid_recognition_loss",
                            val_res["valid_recognition_loss"],
                            self.steps,
                        )
                        self.tb_writer.add_scalar(
                            "valid/wer", val_res["valid_scores"]["wer"], self.steps
                        )
                        self.tb_writer.add_scalars(
                            "valid/wer_scores",
                            val_res["valid_scores"]["wer_scores"],
                            self.steps,
                        )

                    if self.do_translation:
                        self.tb_writer.add_scalar(
                            "valid/valid_translation_loss",
                            val_res["valid_translation_loss"],
                            self.steps,
                        )
                        self.tb_writer.add_scalar(
                            "valid/valid_ppl", val_res["valid_ppl"], self.steps
                        )

                        # Log Scores
                        self.tb_writer.add_scalar(
                            "valid/chrf", val_res["valid_scores"]["chrf"], self.steps
                        )
                        self.tb_writer.add_scalar(
                            "valid/rouge", val_res["valid_scores"]["rouge"], self.steps
                        )
                        self.tb_writer.add_scalar(
                            "valid/bleu", val_res["valid_scores"]["bleu"], self.steps
                        )
                        self.tb_writer.add_scalars(
                            "valid/bleu_scores",
                            val_res["valid_scores"]["bleu_scores"],
                            self.steps,
                        )

                    # =========================================================
                    # PER-LANGUAGE VALIDATION (v4 feature)
                    # =========================================================
                    if self.dev_per_language and self.do_translation:
                        per_lang_bleu = {}
                        per_lang_ppl = {}
                        self.logger.info("=" * 60)
                        self.logger.info("Per-Language Validation Results (Step %d):", self.steps)
                        
                        for lang_name, lang_data in self.dev_per_language.items():
                            try:
                                lang_res = validate_on_data(
                                    model=self.model,
                                    data=lang_data,
                                    batch_size=self.eval_batch_size,
                                    use_cuda=self.use_cuda,
                                    batch_type=self.eval_batch_type,
                                    dataset_version=self.dataset_version,
                                    sgn_dim=self.feature_size,
                                    txt_pad_index=self.txt_pad_index,
                                    do_recognition=False,
                                    recognition_loss_function=None,
                                    recognition_loss_weight=None,
                                    recognition_beam_size=None,
                                    do_translation=True,
                                    translation_loss_function=self.translation_loss_function,
                                    translation_max_output_length=self.translation_max_output_length,
                                    level=self.level,
                                    translation_loss_weight=self.translation_loss_weight,
                                    translation_beam_size=self.eval_translation_beam_size,
                                    translation_beam_alpha=self.eval_translation_beam_alpha,
                                    frame_subsampling_ratio=self.frame_subsampling_ratio,
                                )
                                
                                lang_bleu = lang_res["valid_scores"]["bleu"]
                                lang_ppl_val = lang_res["valid_ppl"]
                                per_lang_bleu[lang_name] = lang_bleu
                                per_lang_ppl[lang_name] = lang_ppl_val
                                
                                # Log to TensorBoard
                                self.tb_writer.add_scalar(
                                    f"valid_lang/{lang_name}/bleu", lang_bleu, self.steps
                                )
                                self.tb_writer.add_scalar(
                                    f"valid_lang/{lang_name}/ppl", lang_ppl_val, self.steps
                                )
                                self.tb_writer.add_scalar(
                                    f"valid_lang/{lang_name}/chrf", 
                                    lang_res["valid_scores"]["chrf"], self.steps
                                )
                                
                                self.logger.info(
                                    "  %s: BLEU-4 %.2f | PPL %.2f | CHRF %.2f",
                                    lang_name.ljust(10), lang_bleu, lang_ppl_val,
                                    lang_res["valid_scores"]["chrf"]
                                )
                            except Exception as e:
                                self.logger.warning("  %s: FAILED - %s", lang_name, str(e))
                                per_lang_bleu[lang_name] = 0.0
                                per_lang_ppl[lang_name] = float('inf')
                        
                        # Compute and log combined metrics (average)
                        if per_lang_bleu:
                            avg_bleu = sum(per_lang_bleu.values()) / len(per_lang_bleu)
                            self.tb_writer.add_scalar("valid/avg_bleu_all_langs", avg_bleu, self.steps)
                            self.logger.info("  %s: BLEU-4 %.2f (average)", "COMBINED".ljust(10), avg_bleu)
                        
                        self.logger.info("=" * 60)
                        self.model.train()

                    if self.early_stopping_metric == "recognition_loss":
                        assert self.do_recognition
                        ckpt_score = val_res["valid_recognition_loss"]
                    elif self.early_stopping_metric == "translation_loss":
                        assert self.do_translation
                        ckpt_score = val_res["valid_translation_loss"]
                    elif self.early_stopping_metric in ["ppl", "perplexity"]:
                        assert self.do_translation
                        ckpt_score = val_res["valid_ppl"]
                    else:
                        ckpt_score = val_res["valid_scores"][self.eval_metric]

                    new_best = False
                    if self.is_best(ckpt_score):
                        self.best_ckpt_score = ckpt_score
                        self.best_all_ckpt_scores = val_res["valid_scores"]
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            "Hooray! New best validation result [%s]!",
                            self.early_stopping_metric,
                        )
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint()

                    if (
                        self.scheduler is not None
                        and self.scheduler_step_at == "validation"
                    ):
                        prev_lr = self.scheduler.optimizer.param_groups[0]["lr"]
                        self.scheduler.step(ckpt_score)
                        now_lr = self.scheduler.optimizer.param_groups[0]["lr"]

                        '''if prev_lr != now_lr:
                            if self.last_best_lr != prev_lr:
                                self.stop = True'''

                    # append to validation report
                    self._add_report(
                        valid_scores=val_res["valid_scores"],
                        valid_recognition_loss=val_res["valid_recognition_loss"]
                        if self.do_recognition
                        else None,
                        valid_translation_loss=val_res["valid_translation_loss"]
                        if self.do_translation
                        else None,
                        valid_ppl=val_res["valid_ppl"] if self.do_translation else None,
                        eval_metric=self.eval_metric,
                        new_best=new_best,
                    )
                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        "Validation result at epoch %3d, step %8d: duration: %.4fs\n\t"
                        "Recognition Beam Size: %d\t"
                        "Translation Beam Size: %d\t"
                        "Translation Beam Alpha: %d\n\t"
                        "Recognition Loss: %4.5f\t"
                        "Translation Loss: %4.5f\t"
                        "PPL: %4.5f\n\t"
                        "Eval Metric: %s\n\t"
                        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
                        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                        "CHRF %.2f\t"
                        "ROUGE %.2f",
                        epoch_no + 1,
                        self.steps,
                        valid_duration,
                        self.eval_recognition_beam_size if self.do_recognition else -1,
                        self.eval_translation_beam_size if self.do_translation else -1,
                        self.eval_translation_beam_alpha if self.do_translation else -1,
                        val_res["valid_recognition_loss"]
                        if self.do_recognition
                        else -1,
                        val_res["valid_translation_loss"]
                        if self.do_translation
                        else -1,
                        val_res["valid_ppl"] if self.do_translation else -1,
                        self.eval_metric.upper(),
                        # WER
                        val_res["valid_scores"]["wer"] if self.do_recognition else -1,
                        val_res["valid_scores"]["wer_scores"]["del_rate"]
                        if self.do_recognition
                        else -1,
                        val_res["valid_scores"]["wer_scores"]["ins_rate"]
                        if self.do_recognition
                        else -1,
                        val_res["valid_scores"]["wer_scores"]["sub_rate"]
                        if self.do_recognition
                        else -1,
                        # BLEU
                        val_res["valid_scores"]["bleu"] if self.do_translation else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu1"]
                        if self.do_translation
                        else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu2"]
                        if self.do_translation
                        else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu3"]
                        if self.do_translation
                        else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu4"]
                        if self.do_translation
                        else -1,
                        # Other
                        val_res["valid_scores"]["chrf"] if self.do_translation else -1,
                        val_res["valid_scores"]["rouge"] if self.do_translation else -1,
                    )

                    # Collect sequence ids for logging and storing, supporting streaming dicts
                    def _collect_sequence_ids(ds):
                        if isinstance(ds, dict):
                            paths = ds["paths"] if isinstance(ds["paths"], list) else [ds["paths"]]
                            ids = []
                            for p in paths:
                                for s in iter_dataset_file(p):
                                    ids.append(s["name"])
                            return ids
                        else:
                            return [s for s in ds.sequence]

                    valid_seq = _collect_sequence_ids(valid_data)

                    self._log_examples(
                        sequences=valid_seq,
                        gls_references=val_res["gls_ref"]
                        if self.do_recognition
                        else None,
                        gls_hypotheses=val_res["gls_hyp"]
                        if self.do_recognition
                        else None,
                        txt_references=val_res["txt_ref"]
                        if self.do_translation
                        else None,
                        txt_hypotheses=val_res["txt_hyp"]
                        if self.do_translation
                        else None,
                    )
                    # store validation set outputs and references
                    if self.do_recognition:
                        self._store_outputs(
                            "dev.hyp.gls", valid_seq, val_res["gls_hyp"], "gls"
                        )
                        self._store_outputs(
                            "references.dev.gls", valid_seq, val_res["gls_ref"]
                        )

                    if self.do_translation:
                        self._store_outputs(
                            "dev.hyp.txt", valid_seq, val_res["txt_hyp"], "txt"
                        )
                        self._store_outputs(
                            "references.dev.txt", valid_seq, val_res["txt_ref"]
                        )

                if self.stop:
                    break

            if self.stop:
                if (
                    self.scheduler is not None
                    and self.scheduler_step_at == "validation"
                    and self.last_best_lr != prev_lr
                ):
                    self.logger.info(
                        "Training ended since there were no improvements in"
                        "the last learning rate step: %f",
                        prev_lr,
                    )
                else:
                    self.logger.info(
                        "Training ended since minimum lr %f was reached.",
                        self.learning_rate_min,
                    )
                break

            self.logger.info(
                "Epoch %3d: Total Training Recognition Loss %.2f "
                " Total Training Translation Loss %.2f ",
                epoch_no + 1,
                epoch_recognition_loss if self.do_recognition else -1,
                epoch_translation_loss if self.do_translation else -1,
            )

        else:
            self.logger.info("Training ended after %3d epochs.", epoch_no + 1)
        self.logger.info(
            "Best validation result at step %8d: %6.2f %s.",
            self.best_ckpt_iteration,
            self.best_ckpt_score,
            self.early_stopping_metric,
        )

        self.tb_writer.close()  # close Tensorboard writer

    def _train_batch(self,  batch: Batch, update: bool = True) -> (Tensor, Tensor):
        """
        Train the model on one batch: Compute the loss, make a gradient step.

        :param batch: training batch
        :param update: if False, only store gradient. if True also make update
        :return normalized_recognition_loss: Normalized recognition loss
        :return normalized_translation_loss: Normalized translation loss
        """

        recognition_loss, translation_loss = self.model.get_loss_for_batch(
            batch=batch,
            recognition_loss_function=self.recognition_loss_function
            if self.do_recognition
            else None,
            translation_loss_function=self.translation_loss_function
            if self.do_translation
            else None,
            recognition_loss_weight=self.recognition_loss_weight
            if self.do_recognition
            else None,
            translation_loss_weight=self.translation_loss_weight
            if self.do_translation
            else None,
        )

        # normalize translation loss
        if self.do_translation:
            if self.translation_normalization_mode == "batch":
                txt_normalization_factor = batch.num_seqs
            elif self.translation_normalization_mode == "tokens":
                txt_normalization_factor = batch.num_txt_tokens
            else:
                raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

            # division needed since loss.backward sums the gradients until updated
            normalized_translation_loss = translation_loss / (
                txt_normalization_factor * self.batch_multiplier
            )
        else:
            normalized_translation_loss = 0

        # TODO (Cihan): Add Gloss Token normalization (?)
        #   I think they are already being normalized by batch
        #   I need to think about if I want to normalize them by token.
        if self.do_recognition:
            normalized_recognition_loss = recognition_loss / self.batch_multiplier
        else:
            normalized_recognition_loss = 0

        total_loss = normalized_recognition_loss + normalized_translation_loss

        # compute gradients
        total_loss.backward()

        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(params=self.model.parameters())

        if update:
            # make gradient step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # increment step counter
            self.steps += 1

        # increment token counter
        if self.do_recognition:
            self.total_gls_tokens += batch.num_gls_tokens
        if self.do_translation:
            self.total_txt_tokens += batch.num_txt_tokens

        return normalized_recognition_loss, normalized_translation_loss

    def _add_report(
        self,
        valid_scores: Dict,
        valid_recognition_loss: float,
        valid_translation_loss: float,
        valid_ppl: float,
        eval_metric: str,
        new_best: bool = False,
    ) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_scores: Dictionary of validation scores
        :param valid_recognition_loss: validation loss (sum over whole validation set)
        :param valid_translation_loss: validation loss (sum over whole validation set)
        :param valid_ppl: validation perplexity
        :param eval_metric: evaluation metric, e.g. "bleu"
        :param new_best: whether this is a new best model
        """
        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group["lr"]

        if new_best:
            self.last_best_lr = current_lr

        if current_lr < self.learning_rate_min:
            self.stop = True

        with open(self.valid_report_file, "a", encoding="utf-8") as opened_file:
            opened_file.write(
                "Steps: {}\t"
                "Recognition Loss: {:.5f}\t"
                "Translation Loss: {:.5f}\t"
                "PPL: {:.5f}\t"
                "Eval Metric: {}\t"
                "WER {:.2f}\t(DEL: {:.2f},\tINS: {:.2f},\tSUB: {:.2f})\t"
                "BLEU-4 {:.2f}\t(BLEU-1: {:.2f},\tBLEU-2: {:.2f},\tBLEU-3: {:.2f},\tBLEU-4: {:.2f})\t"
                "CHRF {:.2f}\t"
                "ROUGE {:.2f}\t"
                "LR: {:.8f}\t{}\n".format(
                    self.steps,
                    valid_recognition_loss if self.do_recognition else -1,
                    valid_translation_loss if self.do_translation else -1,
                    valid_ppl if self.do_translation else -1,
                    eval_metric,
                    # WER
                    valid_scores["wer"] if self.do_recognition else -1,
                    valid_scores["wer_scores"]["del_rate"]
                    if self.do_recognition
                    else -1,
                    valid_scores["wer_scores"]["ins_rate"]
                    if self.do_recognition
                    else -1,
                    valid_scores["wer_scores"]["sub_rate"]
                    if self.do_recognition
                    else -1,
                    # BLEU
                    valid_scores["bleu"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu1"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu2"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu3"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu4"] if self.do_translation else -1,
                    # Other
                    valid_scores["chrf"] if self.do_translation else -1,
                    valid_scores["rouge"] if self.do_translation else -1,
                    current_lr,
                    "*" if new_best else "",
                )
            )

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f"Total params: {n_params:,}")
        trainable_params = [
            n for (n, p) in self.model.named_parameters() if p.requires_grad
        ]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def _log_examples(
        self,
        sequences: List[str],
        gls_references: List[str],
        gls_hypotheses: List[str],
        txt_references: List[str],
        txt_hypotheses: List[str],
    ) -> None:
        """
        Log `self.num_valid_log` number of samples from valid.

        :param sequences: sign video sequence names (list of strings)
        :param txt_hypotheses: decoded txt hypotheses (list of strings)
        :param txt_references: decoded txt references (list of strings)
        :param gls_hypotheses: decoded gls hypotheses (list of strings)
        :param gls_references: decoded gls references (list of strings)
        """

        if self.do_recognition:
            assert len(gls_references) == len(gls_hypotheses)
            num_sequences = len(gls_hypotheses)
        if self.do_translation:
            assert len(txt_references) == len(txt_hypotheses)
            num_sequences = len(txt_hypotheses)

        rand_idx = np.sort(np.random.permutation(num_sequences)[: self.num_valid_log])
        self.logger.info("Logging Recognition and Translation Outputs")
        self.logger.info("=" * 120)
        for ri in rand_idx:
            self.logger.info("Logging Sequence: %s", sequences[ri])
            if self.do_recognition:
                gls_res = wer_single(r=gls_references[ri], h=gls_hypotheses[ri])
                self.logger.info(
                    "\tGloss Reference :\t%s", gls_res["alignment_out"]["align_ref"]
                )
                self.logger.info(
                    "\tGloss Hypothesis:\t%s", gls_res["alignment_out"]["align_hyp"]
                )
                self.logger.info(
                    "\tGloss Alignment :\t%s", gls_res["alignment_out"]["alignment"]
                )
            if self.do_recognition and self.do_translation:
                self.logger.info("\t" + "-" * 116)
            if self.do_translation:
                txt_res = wer_single(r=txt_references[ri], h=txt_hypotheses[ri])
                self.logger.info(
                    "\tText Reference  :\t%s", txt_res["alignment_out"]["align_ref"]
                )
                self.logger.info(
                    "\tText Hypothesis :\t%s", txt_res["alignment_out"]["align_hyp"]
                )
                self.logger.info(
                    "\tText Alignment  :\t%s", txt_res["alignment_out"]["alignment"]
                )
            self.logger.info("=" * 120)

    def _store_outputs(
        self, tag: str, sequence_ids: List[str], hypotheses: List[str], sub_folder=None
    ) -> None:
        """
        Write current validation outputs to file in `self.model_dir.`

        :param hypotheses: list of strings
        """
        if sub_folder:
            out_folder = os.path.join(self.model_dir, sub_folder)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            current_valid_output_file = "{}/{}.{}".format(out_folder, self.steps, tag)
        else:
            out_folder = self.model_dir
            current_valid_output_file = "{}/{}".format(out_folder, tag)

        with open(current_valid_output_file, "w", encoding="utf-8") as opened_file:
            for seq, hyp in zip(sequence_ids, hypotheses):
                opened_file.write("{}|{}\n".format(seq, hyp))


def train(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """
    cfg = load_config(cfg_file)

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    train_data, dev_data, test_data, gls_vocab, txt_vocab = load_data(
        data_cfg=cfg["data"]
    )

    # build model and load parameters into it
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    multimodal = cfg["data"].get("multimodal", 1.0) > 0.0

    model = build_model(
        cfg=cfg["model"],
        multimodal=multimodal,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg)
    
    # =========================================================================
    # LOAD PER-LANGUAGE VALIDATION DATASETS (v4 feature)
    # =========================================================================
    if "dev_per_language" in cfg["data"] and cfg["data"]["dev_per_language"]:
        from torchtext import data as ttdata
        import torch
        
        data_path = cfg["data"].get("data_path", "./data")
        pad_feature_size = (
            sum(cfg["data"]["feature_size"]) 
            if isinstance(cfg["data"]["feature_size"], list)
            else cfg["data"]["feature_size"]
        )
        level = cfg["data"]["level"]
        txt_lowercase = cfg["data"]["txt_lowercase"]
        max_sent_length = cfg["data"]["max_sent_length"]
        
        def tokenize_text(text):
            if level == "char":
                return list(text)
            return text.split()
        
        def tokenize_features(features):
            ft_list = torch.split(features, 1, dim=0)
            return [ft.squeeze() for ft in ft_list]
        
        def stack_features(features, something):
            return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)
        
        sequence_field = ttdata.RawField()
        signer_field = ttdata.RawField()
        sgn_field = ttdata.Field(
            use_vocab=False, init_token=None, dtype=torch.float32,
            preprocessing=tokenize_features,
            tokenize=lambda features: features,
            batch_first=True, include_lengths=True,
            postprocessing=stack_features,
            pad_token=torch.zeros((pad_feature_size,)),
        )
        gls_field = ttdata.Field(
            pad_token=PAD_TOKEN, tokenize=tokenize_text,
            batch_first=True, lower=False, include_lengths=True,
        )
        txt_field = ttdata.Field(
            init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN,
            tokenize=tokenize_text, unk_token=UNK_TOKEN,
            batch_first=True, lower=txt_lowercase, include_lengths=True,
        )
        
        # Share vocabularies with main training
        gls_field.vocab = gls_vocab
        txt_field.vocab = txt_vocab
        
        dev_per_language = {}
        trainer.logger.info("Loading per-language validation datasets...")
        
        for lang_name, lang_file in cfg["data"]["dev_per_language"].items():
            lang_path = os.path.join(data_path, lang_file)
            # Handle glob patterns for .part* files
            if "*" in lang_path:
                import glob as glob_module
                lang_files = sorted(glob_module.glob(lang_path))
            else:
                lang_files = [lang_path]
            
            if not lang_files:
                trainer.logger.warning("No files found for %s: %s", lang_name, lang_path)
                continue
            
            try:
                lang_dataset = SignTranslationDataset(
                    path=lang_files,
                    fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
                    filter_pred=lambda x: len(vars(x)["sgn"]) <= max_sent_length
                    and len(vars(x)["txt"]) <= max_sent_length,
                )
                dev_per_language[lang_name] = lang_dataset
                trainer.logger.info("  %s: %d samples loaded", lang_name, len(lang_dataset))
            except Exception as e:
                trainer.logger.warning("  %s: FAILED to load - %s", lang_name, str(e))
        
        if dev_per_language:
            trainer.set_per_language_validation(dev_per_language)
        else:
            trainer.logger.warning("No per-language validation datasets loaded!")

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")

    # log all entries of config
    log_cfg(cfg, trainer.logger)

    # Prepare a small preview dataset for logging when streaming is enabled
    if isinstance(train_data, dict):
        from torchtext import data as ttdata
        td_paths = train_data["paths"] if isinstance(train_data["paths"], list) else [train_data["paths"]]
        fields = train_data["fields"]
        named_fields = [
            ("sequence", fields[0]),
            ("signer", fields[1]),
            ("sgn", fields[2]),
            ("gls", fields[3]),
            ("txt", fields[4]),
        ]
        preview_samples = []
        try:
            src_iter = iter_dataset_file(td_paths[0])
            for _ in range(16):
                smp = next(src_iter)
                preview_samples.append(smp)
        except StopIteration:
            pass
        examples = [
            ttdata.Example.fromlist(
                [
                    smp["name"],
                    smp["signer"],
                    smp["sign"] + 1e-8,
                    smp["gloss"].strip(),
                    smp["text"].strip(),
                ],
                named_fields,
            )
            for smp in preview_samples
        ]
        preview_dataset = ttdata.Dataset(examples, named_fields) if examples else ttdata.Dataset([], named_fields)
        log_data_info(
            train_data=preview_dataset,
            valid_data=dev_data,
            test_data=test_data,
            gls_vocab=gls_vocab,
            txt_vocab=txt_vocab,
            logging_function=trainer.logger.info,
        )
        del preview_dataset
    else:
        log_data_info(
            train_data=train_data,
            valid_data=dev_data,
            test_data=test_data,
            gls_vocab=gls_vocab,
            txt_vocab=txt_vocab,
            logging_function=trainer.logger.info,
        )

    trainer.logger.info(str(model))

    # store the vocabs
    gls_vocab_file = "{}/gls.vocab".format(cfg["training"]["model_dir"])
    gls_vocab.to_file(gls_vocab_file)
    txt_vocab_file = "{}/txt.vocab".format(cfg["training"]["model_dir"])
    txt_vocab.to_file(txt_vocab_file)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)
    # Delete to speed things up as we don't need training data anymore
    del train_data, dev_data, test_data

    # predict with the best model on validation and test
    # (if test data is available)
    ckpt = "{}/{}.ckpt".format(trainer.model_dir, trainer.best_ckpt_iteration)
    output_name = "best.IT_{:08d}".format(trainer.best_ckpt_iteration)
    output_path = os.path.join(trainer.model_dir, output_name)
    logger = trainer.logger
    del trainer
    test(cfg_file, ckpt=ckpt, output_path=output_path, logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="gpu to run your job on"
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    train(cfg_file=args.config)
