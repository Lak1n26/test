from typing import List

import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer
from src.text_encoder import TextEncoder


class WERMetric(BaseMetric):
    """
    Word Error Rate (WER) metric for ASR.

    WER = (S + D + I) / N

    Where:
    - S = number of substitutions
    - D = number of deletions
    - I = number of insertions
    - N = number of words in reference
    """

    def __init__(
        self,
        text_encoder: TextEncoder = None,
        use_beam_search: bool = False,
        beam_size: int = 10,
        *args,
        **kwargs,
    ):
        """
        Args:
            text_encoder (TextEncoder): Text encoder for decoding predictions.
            use_beam_search (bool): Whether to use beam search decoding.
            beam_size (int): Beam size for beam search.
        """
        super().__init__(*args, **kwargs)
        if text_encoder is None:
            text_encoder = TextEncoder()
        self.text_encoder = text_encoder
        self.use_beam_search = use_beam_search
        self.beam_size = beam_size

    def __call__(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        text: List[str],
        **batch,
    ) -> float:
        """
        Calculate WER for the batch.

        Args:
            log_probs (Tensor): Log probabilities [batch, time, vocab].
            log_probs_length (Tensor): Sequence lengths [batch].
            text (list[str]): Ground truth texts.
        Returns:
            wer (float): Average WER for the batch.
        """
        # Decode predictions
        if self.use_beam_search:
            predictions = self.text_encoder.ctc_beam_search(
                log_probs, log_probs_length, beam_size=self.beam_size
            )
        else:
            predictions = self.text_encoder.ctc_decode(log_probs, log_probs_length)

        # Calculate WER for each pair
        wers = []
        for pred, target in zip(predictions, text):
            wer = calc_wer(target, pred)
            wers.append(wer)

        return sum(wers) / len(wers) if wers else 0.0


class WERMetricBeamSearch(WERMetric):
    """WER metric with beam search decoding."""

    def __init__(self, text_encoder: TextEncoder = None, beam_size: int = 10, **kwargs):
        super().__init__(
            text_encoder=text_encoder,
            use_beam_search=True,
            beam_size=beam_size,
            name=kwargs.pop("name", "WER_beam"),
            **kwargs,
        )

