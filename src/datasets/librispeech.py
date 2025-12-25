import logging
import shutil
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.text_encoder import TextEncoder
from src.transforms import MelSpectrogram
from src.utils.io_utils import ROOT_PATH, read_json, write_json

logger = logging.getLogger(__name__)

# URLs for LibriSpeech dataset parts
LIBRISPEECH_URLS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


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


class LibriSpeechDataset(BaseDataset):
    """
    LibriSpeech dataset for Automatic Speech Recognition.

    Dataset structure after extraction:
        LibriSpeech/
        ├── train-clean-100/
        │   ├── speaker_id/
        │   │   ├── chapter_id/
        │   │   │   ├── speaker_id-chapter_id-utterance_id.flac
        │   │   │   ├── speaker_id-chapter_id.trans.txt
        │   │   │   └── ...
    """

    def __init__(
        self,
        part: str,
        data_dir: str = None,
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
            part (str): dataset partition name (e.g., 'train-clean-100',
                'dev-clean', 'test-clean', 'test-other').
            data_dir (str): path to the data directory. If None, uses
                ROOT_PATH / 'data' / 'datasets' / 'librispeech'.
            text_encoder (TextEncoder): text encoder for converting text
                to indices. If None, creates a default encoder.
            max_audio_length (float): maximum audio length in seconds.
                If provided, filters out longer audios.
            max_text_length (int): maximum text length in characters.
                If provided, filters out longer texts.
            target_sample_rate (int): target sample rate for audio.
            n_mels (int): number of mel filterbanks.
        """
        assert part in LIBRISPEECH_URLS, (
            f"Unknown part: {part}. Available parts: {list(LIBRISPEECH_URLS.keys())}"
        )

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
        else:
            data_dir = Path(data_dir)

        self._data_dir = data_dir
        self._part = part
        self._max_audio_length = max_audio_length
        self._max_text_length = max_text_length
        self._target_sample_rate = target_sample_rate

        # Initialize text encoder
        if text_encoder is None:
            text_encoder = TextEncoder()
        self.text_encoder = text_encoder

        # Initialize mel spectrogram transform
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=target_sample_rate,
            n_mels=n_mels,
        )

        # Download and extract if not exists
        self._prepare_data()

        # Load or create index
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index()
            write_json(index, str(index_path))

        # Filter by audio/text length if needed
        if max_audio_length is not None or max_text_length is not None:
            index = self._filter_by_length(index)

        super().__init__(index, *args, **kwargs)

    def _prepare_data(self):
        """Download and extract the dataset if not already present."""
        part_dir = self._data_dir / "LibriSpeech" / self._part
        if part_dir.exists():
            logger.info(f"Dataset part '{self._part}' already exists at {part_dir}")
            return

        self._data_dir.mkdir(exist_ok=True, parents=True)

        archive_path = self._data_dir / f"{self._part}.tar.gz"

        # Download if archive doesn't exist
        if not archive_path.exists():
            url = LIBRISPEECH_URLS[self._part]
            logger.info(f"Downloading {self._part} from {url}...")
            self._download_file(url, archive_path)

        # Extract
        logger.info(f"Extracting {archive_path}...")
        shutil.unpack_archive(str(archive_path), str(self._data_dir))

        # Optionally remove archive to save space
        # archive_path.unlink()

        logger.info(f"Dataset part '{self._part}' is ready at {part_dir}")

    def _download_file(self, url: str, dest_path: Path):
        """Download a file from URL with progress bar."""
        import urllib.request

        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=dest_path.name
        ) as t:
            urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)

    def _create_index(self):
        """
        Create index for the dataset.

        Parses transcription files and creates a list of dicts with
        audio path, text, and metadata.

        Returns:
            index (list[dict]): list of dataset items.
        """
        index = []
        part_dir = self._data_dir / "LibriSpeech" / self._part

        logger.info(f"Creating index for '{self._part}'...")

        # Walk through all speaker/chapter directories
        for speaker_dir in tqdm(sorted(part_dir.iterdir()), desc="Processing speakers"):
            if not speaker_dir.is_dir():
                continue

            for chapter_dir in sorted(speaker_dir.iterdir()):
                if not chapter_dir.is_dir():
                    continue

                # Find transcription file
                trans_file = list(chapter_dir.glob("*.trans.txt"))
                if not trans_file:
                    continue
                trans_file = trans_file[0]

                # Parse transcription file
                transcriptions = {}
                with open(trans_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            utt_id, text = parts
                            transcriptions[utt_id] = text

                # Process each audio file
                for audio_path in sorted(chapter_dir.glob("*.flac")):
                    utt_id = audio_path.stem
                    if utt_id not in transcriptions:
                        continue

                    text = transcriptions[utt_id]

                    # Get audio info
                    try:
                        num_frames, sample_rate = get_audio_info(str(audio_path))
                        audio_length = num_frames / sample_rate
                    except Exception as e:
                        logger.warning(f"Failed to get audio info for {audio_path}: {e}")
                        continue

                    index.append(
                        {
                            "path": str(audio_path),
                            "text": text.lower(),  # normalize to lowercase
                            "audio_length": audio_length,
                            "speaker_id": speaker_dir.name,
                            "chapter_id": chapter_dir.name,
                            "utterance_id": utt_id,
                        }
                    )

        logger.info(f"Created index with {len(index)} items for '{self._part}'")
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
            if self._max_text_length is not None:
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
        and encode text.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict containing spectrogram,
                encoded text, and metadata.
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

        # Encode text
        text = data_dict["text"]
        text_encoded = self.text_encoder.encode(text)

        instance_data = {
            "audio": audio,
            "spectrogram": spectrogram,
            "text": text,
            "text_encoded": text_encoded,
            "audio_length": data_dict["audio_length"],
            "audio_path": audio_path,
        }

        # Apply instance transforms if any
        instance_data = self.preprocess_data(instance_data)

        return instance_data

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index for ASR task.

        Args:
            index (list[dict]): list of dataset items.
        """
        for entry in index:
            assert "path" in entry, "Each item should include 'path' - path to audio."
            assert "text" in entry, "Each item should include 'text' - transcription."
