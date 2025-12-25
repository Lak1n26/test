from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.text_encoder import TextEncoder
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer class for ASR.

    The class is used to process data without the need of optimizers,
    writers, etc. Required to evaluate the model on the dataset,
    save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        text_encoder: TextEncoder = None,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            text_encoder (TextEncoder): text encoder for decoding.
                If None, creates a default encoder.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # Initialize text encoder
        if text_encoder is None:
            text_encoder = TextEncoder()
        self.text_encoder = text_encoder

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition
        self.save_path = Path(save_path) if save_path else None

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        # Decode predictions and save to disk
        log_probs = batch["log_probs"].detach()
        log_probs_length = batch["log_probs_length"].detach()

        # Get predictions using greedy decoding
        predictions = self.text_encoder.ctc_decode(log_probs, log_probs_length)

        # Get beam search predictions if configured
        beam_size = self.config.inferencer.get("beam_size", None)
        if beam_size is not None and beam_size > 1:
            predictions_beam = self.text_encoder.ctc_beam_search(
                log_probs, log_probs_length, beam_size=beam_size
            )
        else:
            predictions_beam = None

        # Save predictions to disk
        if self.save_path is not None:
            self._save_predictions(
                batch, predictions, predictions_beam, part
            )

        return batch

    def _save_predictions(self, batch, predictions, predictions_beam, part):
        """
        Save predictions to disk.

        Each prediction is saved as a separate text file with the same
        name as the utterance ID.

        Args:
            batch (dict): batch containing metadata.
            predictions (list[str]): greedy decoded predictions.
            predictions_beam (list[str] | None): beam search predictions.
            part (str): partition name.
        """
        save_dir = self.save_path / part
        save_dir.mkdir(exist_ok=True, parents=True)

        # Get utterance IDs if available
        if "utterance_id" in batch:
            utterance_ids = batch["utterance_id"]
        elif "audio_path" in batch:
            utterance_ids = [Path(p).stem for p in batch["audio_path"]]
        else:
            # Fallback: use batch index
            batch_size = len(predictions)
            utterance_ids = [f"sample_{i}" for i in range(batch_size)]

        for i, (utt_id, pred) in enumerate(zip(utterance_ids, predictions)):
            # Save greedy prediction
            pred_path = save_dir / f"{utt_id}.txt"
            with open(pred_path, "w", encoding="utf-8") as f:
                f.write(pred)

            # Save beam search prediction if available
            if predictions_beam is not None:
                beam_path = save_dir / f"{utt_id}_beam.txt"
                with open(beam_path, "w", encoding="utf-8") as f:
                    f.write(predictions_beam[i])

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions.

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """
        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        if self.evaluation_metrics is not None:
            return self.evaluation_metrics.result()
        return {}

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            if tensor_for_device in batch:
                batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch.
        """
        transforms = self.batch_transforms.get("inference")
        if transforms is not None:
            for transform_name in transforms.keys():
                if transform_name in batch:
                    batch[transform_name] = transforms[transform_name](
                        batch[transform_name]
                    )
        return batch
