import logging
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.text_encoder import TextEncoder
from src.transforms import MelSpectrogram
from src.utils.io_utils import read_json, write_json

logger = logging.getLogger(__name__)

# Supported audio extensions
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}


def get_audio_info(audio_path: str):
    """
    Get audio file information using soundfile.

    Args:
        audio_path (str): path to audio file.
    Returns:
        num_frames (int): number of audio frames.
        sample_rate (int): sample rate.
    """
    info = sf.info(audio_path)
    return info.frames, info.samplerate


class CustomDirDataset(BaseDataset):
    """
    Dataset for custom directory structure.

    Expected directory format:
        NameOfTheDirectory/
        ├── audio/
        │   ├── UtteranceID1.wav  (or .flac, .mp3)
        │   ├── UtteranceID2.wav
        │   └── ...
        └── transcriptions/  (optional - may not exist)
            ├── UtteranceID1.txt
            ├── UtteranceID2.txt
            └── ...

    If transcriptions folder doesn't exist, the dataset will work
    in inference-only mode (no ground truth texts).
    """

    def __init__(
        self,
        data_dir: str,
        text_encoder: TextEncoder = None,
        max_audio_length: float = None,
        max_text_length: int = None,
        target_sample_rate: int = 16000,
        n_mels: int = 80,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir (str): path to the data directory containing
                'audio/' and optionally 'transcriptions/' subdirectories.
            text_encoder (TextEncoder): text encoder for converting text
                to indices. If None, creates a default encoder.
            max_audio_length (float): maximum audio length in seconds.
                If provided, filters out longer audios.
            max_text_length (int): maximum text length in characters.
                If provided, filters out longer texts (only if transcriptions exist).
            target_sample_rate (int): target sample rate for audio.
            n_mels (int): number of mel filterbanks.
        """
        data_dir = Path(data_dir)
        assert data_dir.exists(), f"Data directory does not exist: {data_dir}"

        self._data_dir = data_dir
        self._max_audio_length = max_audio_length
        self._max_text_length = max_text_length
        self._target_sample_rate = target_sample_rate

        # Check required subdirectories
        audio_dir = data_dir / "audio"
        assert audio_dir.exists(), (
            f"Audio directory does not exist: {audio_dir}. "
            f"Expected structure: {data_dir}/audio/*.wav"
        )

        transcriptions_dir = data_dir / "transcriptions"
        self._has_transcriptions = transcriptions_dir.exists()

        if not self._has_transcriptions:
            logger.warning(
                f"No transcriptions directory found at {transcriptions_dir}. "
                "Running in inference-only mode."
            )

        # Initialize text encoder
        if text_encoder is None:
            text_encoder = TextEncoder()
        self.text_encoder = text_encoder

        # Initialize mel spectrogram transform
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=target_sample_rate,
            n_mels=n_mels,
        )

        # Load or create index
        index_path = data_dir / "index.json"
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index()
            write_json(index, str(index_path))

        # Filter by audio/text length if needed
        if max_audio_length is not None or max_text_length is not None:
            index = self._filter_by_length(index)

        super().__init__(index, *args, **kwargs)

    def _create_index(self):
        """
        Create index for the dataset.

        Scans audio directory and matches with transcriptions if available.

        Returns:
            index (list[dict]): list of dataset items.
        """
        index = []
        audio_dir = self._data_dir / "audio"
        transcriptions_dir = self._data_dir / "transcriptions"

        logger.info(f"Creating index for custom dataset at {self._data_dir}...")

        # Find all audio files
        audio_files = []
        for ext in AUDIO_EXTENSIONS:
            audio_files.extend(audio_dir.glob(f"*{ext}"))

        audio_files = sorted(audio_files)

        for audio_path in tqdm(audio_files, desc="Processing audio files"):
            utterance_id = audio_path.stem

            # Get transcription if available
            text = None
            if self._has_transcriptions:
                trans_path = transcriptions_dir / f"{utterance_id}.txt"
                if trans_path.exists():
                    with open(trans_path, "r", encoding="utf-8") as f:
                        text = f.read().strip().lower()

            # Get audio info
            try:
                num_frames, sample_rate = get_audio_info(str(audio_path))
                audio_length = num_frames / sample_rate
            except Exception as e:
                logger.warning(f"Failed to load audio info for {audio_path}: {e}")
                continue

            item = {
                "path": str(audio_path),
                "audio_length": audio_length,
                "utterance_id": utterance_id,
            }

            if text is not None:
                item["text"] = text
            elif self._has_transcriptions:
                logger.warning(f"No transcription found for {utterance_id}")

            index.append(item)

        logger.info(f"Created index with {len(index)} items")
        return index

    def _filter_by_length(self, index):
        """
        Filter index by audio and text length.

        Args:
            index (list[dict]): original index.
        Returns:
            filtered_index (list[dict]): filtered index.
        """
        original_len = len(index)
        filtered = []

        for item in index:
            if self._max_audio_length is not None:
                if item["audio_length"] > self._max_audio_length:
                    continue
            if self._max_text_length is not None and "text" in item:
                if len(item["text"]) > self._max_text_length:
                    continue
            filtered.append(item)

        logger.info(
            f"Filtered index from {original_len} to {len(filtered)} items "
            f"(max_audio={self._max_audio_length}s, max_text={self._max_text_length})"
        )
        return filtered

    def __getitem__(self, ind):
        """
        Get element from the index, load audio, compute spectrogram,
        and optionally encode text.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict containing spectrogram,
                text (if available), and metadata.
        """
        data_dict = self._index[ind]

        # Load audio
        audio_path = data_dict["path"]
        audio_np, sr = sf.read(audio_path)
        audio = torch.from_numpy(audio_np).float()
        if audio.ndim > 1:
            audio = audio.mean(dim=-1)  # Convert stereo to mono

        # Resample if necessary
        if sr != self._target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self._target_sample_rate)
            audio = resampler(audio)

        # Compute mel spectrogram
        spectrogram = self.mel_spectrogram(audio)

        instance_data = {
            "audio": audio,
            "spectrogram": spectrogram,
            "audio_length": data_dict["audio_length"],
            "audio_path": audio_path,
            "utterance_id": data_dict["utterance_id"],
        }

        # Add text if available
        if "text" in data_dict:
            text = data_dict["text"]
            instance_data["text"] = text
            instance_data["text_encoded"] = self.text_encoder.encode(text)

        # Apply instance transforms if any
        instance_data = self.preprocess_data(instance_data)

        return instance_data

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index.

        For custom dataset, only 'path' is required.
        'text' is optional (inference mode).

        Args:
            index (list[dict]): list of dataset items.
        """
        for entry in index:
            assert "path" in entry, "Each item should include 'path' - path to audio."
