import os
import typing
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging
from pydub import AudioSegment
import noisereduce as nr
import soundfile as sf

from utils.annotations.audio import Audio

""" Implemented Preprocessors:
- AudioReader - Read audio from path and return audio and label
- WavReader - Read wav file with librosa and return spectrogram and label
- WavSplitter - Split audio into parts based on silence
"""


def import_librosa(object) -> None:
    """Import librosa using importlib"""
    try:
        version = object.librosa.__version__
    except:
        version = "librosa version not found"
        try:
            object.librosa = importlib.import_module('librosa')
            print("librosa version:", object.librosa.__version__)
        except:
            raise ImportError("librosa is required to preprocess Audio. Please install it with `pip install librosa`.")

class AudioReader:
    """ Read audio from path and return audio and label

    Attributes:
        sample_rate (int): Sample rate. Defaults to None.
        log_level (int): Log level. Defaults to logging.INFO.
    """
    def __init__(
            self, 
            sample_rate = None,
            log_level: int = logging.INFO, 
        ) -> None:
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        # import librosa using importlib
        try:
            self.librosa = importlib.import_module('librosa')
            print("librosa version:", self.librosa.__version__)
        except ImportError:
            raise ImportError("librosa is required to augment Audio. Please install it with `pip install librosa`.")

    def __call__(self, audio_path: str, label: typing.Any) -> typing.Tuple[np.ndarray, typing.Any]:
        """ Read audio from path and return audio and label
        
        Args:
            audio_path (str): Path to audio
            label (Any): Label of audio

        Returns:
            Audio: Audio object
            Any: Label of audio
        """
        if isinstance(audio_path, str):
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio {audio_path} not found.")
        else:
            raise TypeError(f"Audio {audio_path} is not a string.")

        audio = Audio(audio_path, sample_rate=self.sample_rate, library=self.librosa)

        if not audio.init_successful:
            audio = None
            self.logger.warning(f"Audio {audio_path} could not be read, returning None.")

        return audio, label
    

class WavReader:
    """Read wav file with librosa and return audio and label
    
    Attributes:
        frame_length (int): Length of the frames in samples.
        frame_step (int): Step size between frames in samples.
        fft_length (int): Number of FFT components.
    """

    def __init__(
            self,
            frame_length: int = 256,
            frame_step: int = 160,
            fft_length: int = 384,
            *args, **kwargs
    ) -> None:
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

        matplotlib.interactive(False)
        # import librosa using importlib
        import_librosa(self)
        
    @staticmethod
    def get_spectrogram(wav_path: str, frame_length: int, frame_step: int, fft_length: int) -> np.ndarray:
        """Compute the spectrogram of a WAV file

        Args:
            wav_path (str): Path to the WAV file.
            frame_length (int): Length of the frames in samples.
            frame_step (int): Step size between frames in samples.
            fft_length (int): Number of FFT components.

        Returns:
            np.ndarray: Spectrogram of the WAV file.
        """
        import_librosa(WavReader)

        # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
        audio, orig_sr = WavReader.librosa.load(wav_path, sr=22050)

        # Compute the Short Time Fourier Transform (STFT) of the audio data and store it in the variable 'spectrogram'
        # The STFT is computed with a hop length of 'frame_step' samples, a window length of 'frame_length' samples, and 'fft_length' FFT components.
        # The resulting spectrogram is also transposed for convenience
        spectrogram = WavReader.librosa.stft(audio, hop_length=frame_step, win_length=frame_length, n_fft=fft_length).T

        # Take the absolute value of the spectrogram to obtain the magnitude spectrum
        spectrogram = np.abs(spectrogram)

        # Take the square root of the magnitude spectrum to obtain the log spectrogram
        spectrogram = np.power(spectrogram, 0.5)

        # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation.
        # A small value of 1e-10 is added to the denominator to prevent division by zero.
        spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-10)

        return spectrogram

    @staticmethod
    def plot_raw_audio(wav_path: str, title: str = None, sr: int = 16000) -> None:
        """Plot the raw audio of a WAV file

        Args:
            wav_path (str): Path to the WAV file.
            sr (int, optional): Sample rate of the WAV file. Defaults to 16000.
            title (str, optional): Title
        """
        import_librosa(WavReader)
        # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
        audio, orig_sr = WavReader.librosa.load(wav_path, sr=sr)

        duration = len(audio) / orig_sr

        time = np.linspace(0, duration, num=len(audio))

        plt.figure(figsize=(15, 5))
        plt.plot(time, audio)
        plt.title(title) if title else plt.title("Audio Plot")
        plt.ylabel("signal wave")
        plt.xlabel("time (s)")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_spectrogram(spectrogram: np.ndarray, title:str = "", transpose: bool = True, invert: bool = True) -> None:
        """Plot the spectrogram of a WAV file

        Args:
            spectrogram (np.ndarray): Spectrogram of the WAV file.
            title (str, optional): Title of the plot. Defaults to None.
            transpose (bool, optional): Transpose the spectrogram. Defaults to True.
            invert (bool, optional): Invert the spectrogram. Defaults to True.
        """
        if transpose:
            spectrogram = spectrogram.T
        
        if invert:
            spectrogram = spectrogram[::-1]

        plt.figure(figsize=(15, 5))
        plt.imshow(spectrogram, aspect="auto", origin="lower")
        plt.title(f"Spectrogram: {title}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def __call__(self, audio_path: str, label: typing.Any):
        """
        Extract the spectrogram and label of a WAV file.

        Args:
            audio_path (str): Path to the WAV file.
            label (typing.Any): Label of the WAV file.

        Returns:
            Tuple[np.ndarray, typing.Any]: Spectrogram of the WAV file and its label.
        """
        return self.get_spectrogram(audio_path, self.frame_length, self.frame_step, self.fft_length), label


class WavSplitter:
    """Split audio into parts based on silence

    Attributes:
        max_duration (int): Maximum duration of each part in milliseconds.
        threshold (int): Threshold for silence detection in dBFS.
    """

    def __init__(self, max_duration: int = 8000, threshold: int = -50):
        self.max_duration = max_duration
        self.threshold = threshold

    def check_audio_duration(self, audio_name: str) -> bool:
        """Check if audio duration is greater than maximum duration

        Args:
            audio_name (str): Name of the audio file.
        """
        # Load the audio file
        audio = AudioSegment.from_file(f"data/input/{audio_name}", format="wav")
        return len(audio) > self.max_duration

    def split_audio(self, audio_name: str) -> typing.List[AudioSegment]:
        """Split audio into parts based on silence

        Args:
            audio_name (str): Name of the audio file.

        Returns:
            typing.List[AudioSegment]: List of paths to the split audio files.
        """

        # Load the audio file
        audio = AudioSegment.from_file(f"data/input/{audio_name}", format="wav")
        
        # Find periods of silence
        silences = [0] + [i for i, sample in enumerate(audio) if sample.dBFS < self.threshold] + [len(audio)]

        parts = []
        start = 0

        # Split the audio into parts based on the periods of silence
        for i in range(1, len(silences)):
            if (silences[i] - start) > self.max_duration:
                part = audio[start:silences[i-1]]
                parts.append(part)
                start = silences[i-1]

        # Add the last part
        if start < len(audio):
            parts.append(audio[start:])

        # Remove the extension from the audio name
        audio_name = audio_name.split(".")[0]

        # Check if exist temp in the audio name
        if "temp" in audio_name:
            input_path = f"data/input{audio_name}"
        else:
            input_path = f"data/input/temp/{audio_name}"

        # Create a temporary directory to store the split audio files
        os.makedirs(input_path, exist_ok=True)

        # Save the parts
        for idx, part in enumerate(parts):
            part.export(f"{input_path}/part_{idx+1}.wav", format="wav")

        return [f"{input_path}/part_{idx+1}.wav" for idx in range(len(parts))]

    def __call__(self, audio_name: str) -> typing.List[str]:
        """ Check audio duration and split audio if necessary

        Args:
            audio_name (str): Name of the audio file.

        Returns:
            typing.List[str]: List of paths to the split audio files.
        """    
    
        if self.check_audio_duration(audio_name):
            return self.split_audio(audio_name)
        
        return [audio_name]


class WavEnhancer:
    """Reduce noise and normalize audio"""

    def __init__(self):
        # import librosa using importlib
        import_librosa(self)

    def reduce_noise(self, audio_name: str) -> np.ndarray:
        """Reduce noise from audio using noise profile

        Args:
            audio_name (str): Name of the audio file.

        Returns:
            np.ndarray: Audio with reduced noise.
        """
        import_librosa(WavEnhancer)
        # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'sr '
        audio_path = f"data/input/{audio_name}"
        audio, sr = WavEnhancer.librosa.load(audio_path)

        # reduce noise from audio using noisereduce
        audio_denoised = nr.reduce_noise(y=audio, sr=sr)

        return audio_denoised, sr
    
    def normalize_audio(self, audio_denoised: np.ndarray, sr: int) -> np.ndarray:
        """Normalize audio

        Args:
            audio_name (str): Name of the audio file.
            sr (int): Sample rate of the audio.

        Returns:
            np.ndarray: Normalized audio.
        """
        normalized_audio = audio_denoised / max(abs(audio_denoised))

        return normalized_audio

    def __call__(self, audio_name: str) -> str:
        """Reduce noise and normalize audio

        Args:
            audio_name (str): Name of the audio file.

        Returns:
            str: Path to the audio file.
        """
        audio_denoised, sr = self.reduce_noise(audio_name)
        audio_normalized = self.normalize_audio(audio_denoised, sr)

        audio_path = f"/temp/{audio_name}"

        sf.write(f"data/input/{audio_name}", audio_normalized, sr)

        return audio_path


class PreProcessors:
    """Preprocess the audio data using the WavEnhancer and WavSplitter preprocessors."""

    def __call__(self, audio_name: str) -> typing.List[str]:
        """Preprocess the audio data.

        Args:
            audio_name (str): Name of the audio file.
        Returns:
            typing.List[str]: List of paths to the split audio files after preprocessing.
        """
        
        wav_enhancer = WavEnhancer()
        audio_path = wav_enhancer(audio_name)

        splitter = WavSplitter()
        audios = splitter(audio_path)

        return audios
        