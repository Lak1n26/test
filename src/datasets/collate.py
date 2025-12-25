import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch for ASR training.

    Handles variable-length audio spectrograms and text sequences
    by padding them to the maximum length in the batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__. Each item should contain:
            - 'spectrogram' (Tensor): mel spectrogram [n_mels, time]
            - 'text_encoded' (Tensor): encoded text sequence [seq_len]
            - 'text' (str): original text
            - 'audio_path' (str): path to audio file
            And optionally other metadata.

    Returns:
        result_batch (dict[Tensor]): dict containing:
            - 'spectrogram' (Tensor): padded spectrograms [batch, n_mels, max_time]
            - 'spectrogram_length' (Tensor): original lengths [batch]
            - 'text_encoded' (Tensor): padded text indices [batch, max_seq_len]
            - 'text_encoded_length' (Tensor): original text lengths [batch]
            - 'text' (list[str]): original texts
            - 'audio_path' (list[str]): paths to audio files
    """
    result_batch = {}

    # Process spectrograms - pad along time dimension
    # Spectrograms have shape [n_mels, time], we need to pad time dimension
    spectrograms = [item["spectrogram"].transpose(0, 1) for item in dataset_items]
    # After transpose: [time, n_mels]
    spectrogram_lengths = torch.tensor([s.shape[0] for s in spectrograms])

    # Pad sequences (pads along first dimension which is time)
    padded_spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=0)
    # Shape: [batch, max_time, n_mels]
    # Transpose back to [batch, n_mels, max_time]
    result_batch["spectrogram"] = padded_spectrograms.transpose(1, 2)
    result_batch["spectrogram_length"] = spectrogram_lengths

    # Process text - pad encoded text sequences
    if "text_encoded" in dataset_items[0]:
        text_encoded = [item["text_encoded"] for item in dataset_items]
        text_encoded_lengths = torch.tensor([t.shape[0] for t in text_encoded])

        padded_text = pad_sequence(text_encoded, batch_first=True, padding_value=0)
        result_batch["text_encoded"] = padded_text
        result_batch["text_encoded_length"] = text_encoded_lengths

    # Collect string fields as lists
    if "text" in dataset_items[0]:
        result_batch["text"] = [item["text"] for item in dataset_items]

    if "audio_path" in dataset_items[0]:
        result_batch["audio_path"] = [item["audio_path"] for item in dataset_items]

    # Collect optional metadata
    if "utterance_id" in dataset_items[0]:
        result_batch["utterance_id"] = [item["utterance_id"] for item in dataset_items]

    return result_batch
