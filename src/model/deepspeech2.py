import math

import torch
from torch import nn


class MaskConv2d(nn.Module):
    """
    Masked 2D convolution that properly handles variable-length inputs.

    Applies masking based on sequence lengths after convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        Args:
            x (Tensor): Input tensor [batch, channels, freq, time].
            lengths (Tensor): Sequence lengths [batch].
        Returns:
            output (Tensor): Convolved tensor.
            new_lengths (Tensor): Updated sequence lengths.
        """
        output = self.conv(x)

        # Calculate new lengths after convolution
        kernel_size = self.conv.kernel_size[1]
        stride = self.conv.stride[1]
        padding = self.conv.padding[1]

        new_lengths = (lengths + 2 * padding - kernel_size) // stride + 1
        new_lengths = new_lengths.clamp(min=1).to(output.device)

        # Create mask and apply
        batch_size, channels, freq, time = output.shape
        mask = torch.arange(time, device=output.device).expand(batch_size, time)
        mask = mask < new_lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, time]

        output = output * mask

        return output, new_lengths


class BatchRNN(nn.Module):
    """
    Bidirectional RNN layer with batch normalization.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "gru",
        bidirectional: bool = True,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.batch_norm = batch_norm

        if batch_norm:
            self.bn = nn.BatchNorm1d(input_size)

        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        Args:
            x (Tensor): Input tensor [batch, time, features].
            lengths (Tensor): Sequence lengths [batch].
        Returns:
            output (Tensor): RNN output [batch, time, hidden * directions].
        """
        if self.batch_norm:
            # Apply batch norm: need to transpose for BatchNorm1d
            x = x.transpose(1, 2)  # [batch, features, time]
            x = self.bn(x)
            x = x.transpose(1, 2)  # [batch, time, features]

        # Pack padded sequence
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        output_packed, _ = self.rnn(x_packed)

        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)

        return output


class DeepSpeech2(nn.Module):
    """
    DeepSpeech2 model for ASR with CTC loss.

    Architecture:
    1. 2-3 convolutional layers for feature extraction
    2. 5-7 bidirectional RNN layers (GRU or LSTM)
    3. Fully connected layer for character prediction

    Paper: https://arxiv.org/abs/1512.02595
    """

    def __init__(
        self,
        n_feats: int = 80,
        n_tokens: int = 29,
        n_conv_layers: int = 2,
        conv_channels: int = 32,
        n_rnn_layers: int = 5,
        rnn_hidden_size: int = 512,
        rnn_type: str = "gru",
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_feats (int): Number of input features (mel bins).
            n_tokens (int): Number of output tokens (vocab size including blank).
            n_conv_layers (int): Number of convolutional layers.
            conv_channels (int): Number of channels in conv layers.
            n_rnn_layers (int): Number of RNN layers.
            rnn_hidden_size (int): Hidden size of RNN layers.
            rnn_type (str): Type of RNN ('gru' or 'lstm').
            bidirectional (bool): Whether to use bidirectional RNN.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.n_feats = n_feats
        self.n_tokens = n_tokens

        # Convolutional layers
        self.conv_layers = nn.ModuleList()

        # First conv layer
        self.conv_layers.append(
            MaskConv2d(
                in_channels=1,
                out_channels=conv_channels,
                kernel_size=(41, 11),
                stride=(2, 2),
                padding=(20, 5),
            )
        )

        # Second conv layer
        if n_conv_layers >= 2:
            self.conv_layers.append(
                MaskConv2d(
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=(21, 11),
                    stride=(2, 1),
                    padding=(10, 5),
                )
            )

        # Third conv layer (optional)
        if n_conv_layers >= 3:
            self.conv_layers.append(
                MaskConv2d(
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=(21, 11),
                    stride=(2, 1),
                    padding=(10, 5),
                )
            )

        self.conv_bn = nn.BatchNorm2d(conv_channels)
        self.conv_activation = nn.Hardtanh(0, 20, inplace=True)

        # Calculate RNN input size after convolutions
        rnn_input_size = self._get_conv_output_size(n_feats, n_conv_layers) * conv_channels

        # RNN layers
        self.rnn_layers = nn.ModuleList()
        rnn_output_size = rnn_hidden_size * (2 if bidirectional else 1)

        for i in range(n_rnn_layers):
            input_size = rnn_input_size if i == 0 else rnn_output_size
            self.rnn_layers.append(
                BatchRNN(
                    input_size=input_size,
                    hidden_size=rnn_hidden_size,
                    rnn_type=rnn_type,
                    bidirectional=bidirectional,
                    batch_norm=(i > 0),  # No batch norm on first layer
                )
            )

        self.dropout = nn.Dropout(dropout)

        # Final fully connected layer
        self.fc = nn.Linear(rnn_output_size, n_tokens)

    def _get_conv_output_size(self, n_feats: int, n_conv_layers: int) -> int:
        """Calculate the frequency dimension after conv layers."""
        size = n_feats
        # First conv: kernel=41, stride=2, padding=20
        size = (size + 2 * 20 - 41) // 2 + 1
        # Second conv: kernel=21, stride=2, padding=10
        if n_conv_layers >= 2:
            size = (size + 2 * 10 - 21) // 2 + 1
        # Third conv: kernel=21, stride=2, padding=10
        if n_conv_layers >= 3:
            size = (size + 2 * 10 - 21) // 2 + 1
        return size

    def forward(
        self,
        spectrogram: torch.Tensor,
        spectrogram_length: torch.Tensor,
        **batch,
    ) -> dict:
        """
        Forward pass.

        Args:
            spectrogram (Tensor): Input spectrogram [batch, n_mels, time].
            spectrogram_length (Tensor): Lengths of spectrograms [batch].
        Returns:
            output (dict): Dictionary with 'log_probs' and 'log_probs_length'.
        """
        # Add channel dimension: [batch, 1, n_mels, time]
        x = spectrogram.unsqueeze(1)
        lengths = spectrogram_length.clone()

        # Convolutional layers
        for conv in self.conv_layers:
            x, lengths = conv(x, lengths)

        x = self.conv_bn(x)
        x = self.conv_activation(x)

        # Reshape for RNN: [batch, channels, freq, time] -> [batch, time, channels * freq]
        batch_size, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)  # [batch, time, channels, freq]
        x = x.reshape(batch_size, time, channels * freq)

        # RNN layers
        for rnn in self.rnn_layers:
            x = rnn(x, lengths)
            x = self.dropout(x)

        # Fully connected layer
        logits = self.fc(x)  # [batch, time, n_tokens]

        # Log softmax for CTC
        log_probs = torch.log_softmax(logits, dim=-1)

        return {
            "log_probs": log_probs,
            "log_probs_length": lengths,
        }

    def transform_input_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculate output lengths after convolutional layers.

        Useful for CTC loss which needs output sequence lengths.
        """
        lengths = input_lengths.clone()
        for conv in self.conv_layers:
            kernel_size = conv.conv.kernel_size[1]
            stride = conv.conv.stride[1]
            padding = conv.conv.padding[1]
            lengths = (lengths + 2 * padding - kernel_size) // stride + 1
        return lengths.clamp(min=1)

    def __str__(self):
        """Model summary with parameter count."""
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters:,}"
        result_info += f"\nTrainable parameters: {trainable_parameters:,}"

        return result_info

