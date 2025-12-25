import torch
from torch import nn


class CTCLoss(nn.Module):
    """
    CTC (Connectionist Temporal Classification) Loss for ASR.

    Wraps PyTorch's CTCLoss with proper handling of input/target lengths.
    """

    def __init__(self, blank: int = 0, zero_infinity: bool = True):
        """
        Args:
            blank (int): Index of the blank token.
            zero_infinity (bool): Whether to zero infinite losses and gradients.
        """
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity)

    def forward(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        text_encoded: torch.Tensor,
        text_encoded_length: torch.Tensor,
        **batch,
    ) -> dict:
        """
        Calculate CTC loss.

        Args:
            log_probs (Tensor): Log probabilities from model,
                shape [batch, time, n_tokens].
            log_probs_length (Tensor): Lengths of log_probs sequences,
                shape [batch].
            text_encoded (Tensor): Encoded target text,
                shape [batch, max_target_len].
            text_encoded_length (Tensor): Lengths of target sequences,
                shape [batch].
        Returns:
            loss_dict (dict): Dictionary with 'loss' key.
        """
        # CTC expects log_probs in shape [time, batch, n_tokens]
        log_probs_transposed = log_probs.transpose(0, 1)

        # Ensure lengths are on CPU and int32
        input_lengths = log_probs_length.cpu().to(torch.int32)
        target_lengths = text_encoded_length.cpu().to(torch.int32)

        # Calculate loss
        loss = self.ctc_loss(
            log_probs_transposed,
            text_encoded,
            input_lengths,
            target_lengths,
        )

        return {"loss": loss}


class CTCLossWithLabelSmoothing(nn.Module):
    """
    CTC Loss with label smoothing for regularization.

    Adds a small probability mass to non-target tokens to prevent
    over-confident predictions.
    """

    def __init__(
        self,
        blank: int = 0,
        zero_infinity: bool = True,
        smoothing: float = 0.1,
    ):
        """
        Args:
            blank (int): Index of the blank token.
            zero_infinity (bool): Whether to zero infinite losses.
            smoothing (float): Label smoothing factor.
        """
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity)
        self.smoothing = smoothing

    def forward(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        text_encoded: torch.Tensor,
        text_encoded_length: torch.Tensor,
        **batch,
    ) -> dict:
        """
        Calculate CTC loss with label smoothing.
        """
        # Standard CTC loss
        log_probs_transposed = log_probs.transpose(0, 1)
        input_lengths = log_probs_length.cpu().to(torch.int32)
        target_lengths = text_encoded_length.cpu().to(torch.int32)

        ctc_loss = self.ctc_loss(
            log_probs_transposed,
            text_encoded,
            input_lengths,
            target_lengths,
        )

        # KL divergence with uniform distribution for smoothing
        # This encourages the model to not be over-confident
        n_tokens = log_probs.shape[-1]
        uniform_loss = -log_probs.mean()

        # Combined loss
        loss = (1 - self.smoothing) * ctc_loss + self.smoothing * uniform_loss

        return {
            "loss": loss,
            "ctc_loss": ctc_loss,
            "smooth_loss": uniform_loss,
        }

