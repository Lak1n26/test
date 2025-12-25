import random

import torch
import pandas as pd
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder import TextEncoder
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class for ASR. Defines the logic of batch logging and processing.
    """

    def __init__(self, *args, text_encoder: TextEncoder = None, **kwargs):
        """
        Initialize the trainer.

        Args:
            text_encoder (TextEncoder): text encoder for decoding predictions.
                If None, creates a default encoder.
        """
        super().__init__(*args, **kwargs)
        if text_encoder is None:
            text_encoder = TextEncoder()
        self.text_encoder = text_encoder

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if self.writer is None:
            return

        # Number of examples to log
        n_examples = min(3, batch["spectrogram"].shape[0])

        if mode == "train":
            # Log spectrograms (only during training, every log_step)
            self._log_spectrogram(batch, n_examples)

            # Log audio samples (less frequently - every 10 log steps)
            if batch_idx % (self.log_step * 10) == 0:
                self._log_audio(batch, n_examples=1)

            # Log text predictions
            self._log_predictions(batch, n_examples)

        else:
            # During evaluation, log more examples
            self._log_spectrogram(batch, n_examples)
            self._log_audio(batch, n_examples=1)
            self._log_predictions(batch, n_examples)

    def _log_spectrogram(self, batch, n_examples):
        """
        Log spectrograms as images.

        Args:
            batch (dict): batch containing spectrograms.
            n_examples (int): number of examples to log.
        """
        spectrograms = batch["spectrogram"][:n_examples].detach().cpu()

        for i, spec in enumerate(spectrograms):
            # spec shape: [n_mels, time]
            # Normalize to [0, 255] for wandb
            spec_normalized = spec - spec.min()
            if spec_normalized.max() > 0:
                spec_normalized = spec_normalized / spec_normalized.max() * 255
            self.writer.add_image(f"spectrogram_{i}", spec_normalized.unsqueeze(0))

    def _log_audio(self, batch, n_examples=1):
        """
        Log audio samples.

        Args:
            batch (dict): batch containing audio paths or waveforms.
            n_examples (int): number of examples to log.
        """
        # Log from audio paths if available
        if "audio" in batch:
            audios = batch["audio"][:n_examples]
            sample_rate = 16000  # Default sample rate

            for i, audio in enumerate(audios):
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu()
                    self.writer.add_audio(f"audio_{i}", audio, sample_rate)

    def _log_predictions(self, batch, n_examples):
        """
        Log text predictions with ground truth and metrics.

        Args:
            batch (dict): batch containing predictions and targets.
            n_examples (int): number of examples to log.
        """
        log_probs = batch["log_probs"][:n_examples].detach()
        log_probs_length = batch["log_probs_length"][:n_examples].detach()
        texts = batch["text"][:n_examples]

        # Decode predictions
        predictions = self.text_encoder.ctc_decode(log_probs, log_probs_length)

        # Build log table
        table_data = []
        for i, (pred, target) in enumerate(zip(predictions, texts)):
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            row = {
                "index": i,
                "target": target,
                "prediction": pred,
                "WER": f"{wer:.2f}%",
                "CER": f"{cer:.2f}%",
            }
            table_data.append(row)

        # Log as table
        if hasattr(self.writer, "add_table"):
            
            df = pd.DataFrame(table_data)
            self.writer.add_table("predictions", df)
        else:
            # Fallback: log as text
            text_log = "\n".join([
                f"[{r['index']}] WER: {r['WER']}, CER: {r['CER']}\n"
                f"  Target: {r['target']}\n"
                f"  Pred:   {r['prediction']}"
                for r in table_data
            ])
            self.writer.add_text("predictions", text_log)

        # Also log random prediction as scalar-like text for quick view
        if len(predictions) > 0:
            idx = random.randint(0, len(predictions) - 1)
            sample_text = (
                f"Target: {texts[idx]}\n"
                f"Pred: {predictions[idx]}"
            )
            self.writer.add_text("sample_prediction", sample_text)
