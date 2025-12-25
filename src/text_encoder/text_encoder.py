import re
from typing import List, Union

import torch


class TextEncoder:
    """
    Text encoder for ASR.

    Converts text to indices and back. Supports CTC decoding
    with blank token handling.

    Default alphabet includes lowercase English letters, space,
    and common punctuation. The blank token (for CTC) is at index 0.
    """

    # Default alphabet for English ASR
    ALPHABET = list(" " + "abcdefghijklmnopqrstuvwxyz" + "'")

    def __init__(self, alphabet: List[str] = None):
        """
        Args:
            alphabet (list[str]): List of characters in the vocabulary.
                If None, uses default English alphabet.
                The blank token is automatically added at index 0.
        """
        if alphabet is None:
            alphabet = self.ALPHABET.copy()

        self.blank_token = "<blank>"
        self.unk_token = "<unk>"

        # Build vocabulary: blank at 0, then alphabet, then unk
        self._vocab = [self.blank_token] + alphabet + [self.unk_token]

        # Mappings
        self._char2idx = {char: idx for idx, char in enumerate(self._vocab)}
        self._idx2char = {idx: char for idx, char in enumerate(self._vocab)}

        self.blank_idx = 0
        self.unk_idx = len(self._vocab) - 1

    def __len__(self):
        """Return vocabulary size (including blank and unk)."""
        return len(self._vocab)

    @property
    def vocab_size(self):
        """Return vocabulary size (including blank and unk)."""
        return len(self._vocab)

    @property
    def alphabet(self):
        """Return the alphabet (without blank and unk)."""
        return self._vocab[1:-1]

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text string to tensor of indices.

        Args:
            text (str): Input text string.
        Returns:
            encoded (Tensor): Tensor of indices with shape [seq_len].
        """
        text = self.normalize_text(text)
        indices = []
        for char in text:
            if char in self._char2idx:
                indices.append(self._char2idx[char])
            else:
                indices.append(self.unk_idx)
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices: Union[torch.Tensor, List[int]]) -> str:
        """
        Decode indices to text string.

        Args:
            indices (Tensor | list[int]): Sequence of indices.
        Returns:
            text (str): Decoded text string.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        chars = []
        for idx in indices:
            if idx == self.blank_idx:
                continue  # Skip blank tokens
            if idx == self.unk_idx:
                chars.append("?")  # Replace unknown with ?
            elif idx in self._idx2char:
                chars.append(self._idx2char[idx])

        return "".join(chars)

    def ctc_decode(
        self, log_probs: torch.Tensor, log_probs_length: torch.Tensor = None
    ) -> List[str]:
        """
        Greedy CTC decoding.

        Takes argmax at each timestep and removes duplicates and blanks.

        Args:
            log_probs (Tensor): Log probabilities from model,
                shape [batch, time, vocab_size] or [time, vocab_size].
            log_probs_length (Tensor): Lengths of each sequence in batch,
                shape [batch]. If None, uses full length.
        Returns:
            texts (list[str]): List of decoded texts.
        """
        if log_probs.dim() == 2:
            log_probs = log_probs.unsqueeze(0)
            if log_probs_length is not None:
                log_probs_length = log_probs_length.unsqueeze(0)

        batch_size = log_probs.shape[0]
        if log_probs_length is None:
            log_probs_length = torch.full(
                (batch_size,), log_probs.shape[1], dtype=torch.long
            )

        # Get argmax predictions
        predictions = log_probs.argmax(dim=-1)  # [batch, time]

        texts = []
        for i in range(batch_size):
            length = log_probs_length[i].item()
            pred = predictions[i, :length].tolist()

            # Remove consecutive duplicates and blanks
            decoded = []
            prev_idx = None
            for idx in pred:
                if idx != prev_idx:
                    if idx != self.blank_idx:
                        decoded.append(idx)
                prev_idx = idx

            text = self.decode(decoded)
            texts.append(text)

        return texts

    def ctc_beam_search(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor = None,
        beam_size: int = 10,
    ) -> List[str]:
        """
        Beam search CTC decoding.

        Args:
            log_probs (Tensor): Log probabilities from model,
                shape [batch, time, vocab_size] or [time, vocab_size].
            log_probs_length (Tensor): Lengths of each sequence in batch.
            beam_size (int): Number of beams to keep.
        Returns:
            texts (list[str]): List of decoded texts (best hypothesis).
        """
        if log_probs.dim() == 2:
            log_probs = log_probs.unsqueeze(0)
            if log_probs_length is not None:
                log_probs_length = log_probs_length.unsqueeze(0)

        batch_size = log_probs.shape[0]
        if log_probs_length is None:
            log_probs_length = torch.full(
                (batch_size,), log_probs.shape[1], dtype=torch.long
            )

        texts = []
        for i in range(batch_size):
            length = log_probs_length[i].item()
            probs = log_probs[i, :length]  # [time, vocab_size]

            # Beam search for single sequence
            text = self._beam_search_single(probs, beam_size)
            texts.append(text)

        return texts

    def _beam_search_single(
        self, log_probs: torch.Tensor, beam_size: int
    ) -> str:
        """
        Beam search for a single sequence.

        Args:
            log_probs (Tensor): Log probabilities, shape [time, vocab_size].
            beam_size (int): Number of beams.
        Returns:
            text (str): Best decoded text.
        """
        # Beam: (prefix_tuple, last_char, log_prob)
        # We track both the output prefix and whether last emission was blank
        # State: {prefix: (prob_blank, prob_non_blank)}

        T, V = log_probs.shape
        log_probs = log_probs.cpu()

        # Initialize with empty prefix
        # beams: dict mapping prefix (tuple) -> (prob_with_blank, prob_without_blank)
        beams = {(): (0.0, float("-inf"))}  # Start with blank probability = 1

        for t in range(T):
            new_beams = {}

            for prefix, (pb, pnb) in beams.items():
                # Total probability for this prefix
                p_total = self._log_add(pb, pnb)

                # Extend with blank
                new_pb = p_total + log_probs[t, self.blank_idx].item()
                if prefix in new_beams:
                    new_beams[prefix] = (
                        self._log_add(new_beams[prefix][0], new_pb),
                        new_beams[prefix][1],
                    )
                else:
                    new_beams[prefix] = (new_pb, float("-inf"))

                # Extend with each character
                for c in range(1, V):
                    if c == self.unk_idx:
                        continue  # Skip unknown token

                    c_prob = log_probs[t, c].item()

                    if len(prefix) > 0 and prefix[-1] == c:
                        # Same character as last - only extend from blank
                        new_pnb = pb + c_prob
                    else:
                        # Different character - extend from both
                        new_pnb = p_total + c_prob

                    new_prefix = prefix + (c,)

                    if new_prefix in new_beams:
                        new_beams[new_prefix] = (
                            new_beams[new_prefix][0],
                            self._log_add(new_beams[new_prefix][1], new_pnb),
                        )
                    else:
                        new_beams[new_prefix] = (float("-inf"), new_pnb)

                    # Also handle case where same char repeats (from non-blank)
                    if len(prefix) > 0 and prefix[-1] == c:
                        repeat_pnb = pnb + c_prob
                        new_beams[new_prefix] = (
                            new_beams[new_prefix][0],
                            self._log_add(new_beams[new_prefix][1], repeat_pnb),
                        )

            # Prune to top beam_size
            beam_list = [
                (prefix, self._log_add(pb, pnb))
                for prefix, (pb, pnb) in new_beams.items()
            ]
            beam_list.sort(key=lambda x: x[1], reverse=True)
            beam_list = beam_list[:beam_size]

            beams = {}
            for prefix, _ in beam_list:
                beams[prefix] = new_beams[prefix]

        # Get best beam
        best_prefix = max(beams.keys(), key=lambda p: self._log_add(*beams[p]))
        text = self.decode(list(best_prefix))

        return text

    @staticmethod
    def _log_add(a: float, b: float) -> float:
        """Numerically stable log addition."""
        if a == float("-inf"):
            return b
        if b == float("-inf"):
            return a
        if a > b:
            return a + torch.log1p(torch.exp(torch.tensor(b - a))).item()
        else:
            return b + torch.log1p(torch.exp(torch.tensor(a - b))).item()

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for encoding.

        Converts to lowercase and removes unsupported characters.

        Args:
            text (str): Input text.
        Returns:
            normalized (str): Normalized text.
        """
        text = text.lower()
        # Keep only alphanumeric, spaces, and apostrophes
        text = re.sub(r"[^a-z '\"]", " ", text)
        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)
        # Remove quotes, keep apostrophes
        text = text.replace('"', "")
        text = text.strip()
        return text

