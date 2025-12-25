from typing import List

import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer
from src.text_encoder import TextEncoder


class CERMetric(BaseMetric):
    """
    Character Error Rate (CER) metric for ASR.

    CER = (S + D + I) / N

    Where:
    - S = number of character substitutions
    - D = number of character deletions
    - I = number of character insertions
    - N = number of characters in reference
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
        Calculate CER for the batch.

        Args:
            log_probs (Tensor): Log probabilities [batch, time, vocab].
            log_probs_length (Tensor): Sequence lengths [batch].
            text (list[str]): Ground truth texts.
        Returns:
            cer (float): Average CER for the batch.
        """
        # Decode predictions
        if self.use_beam_search:
            predictions = self.text_encoder.ctc_beam_search(
                log_probs, log_probs_length, beam_size=self.beam_size
            )
        else:
            predictions = self.text_encoder.ctc_decode(log_probs, log_probs_length)

        # Calculate CER for each pair
        cers = []
        for pred, target in zip(predictions, text):
            cer = calc_cer(target, pred)
            cers.append(cer)

        return sum(cers) / len(cers) if cers else 0.0


class CERMetricBeamSearch(CERMetric):
    """CER metric with beam search decoding."""

    def __init__(self, text_encoder: TextEncoder = None, beam_size: int = 10, **kwargs):
        super().__init__(
            text_encoder=text_encoder,
            use_beam_search=True,
            beam_size=beam_size,
            name=kwargs.pop("name", "CER_beam"),
            **kwargs,
        )

