import os
import typing
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging
import datetime
from pydub import AudioSegment

from utils.annotations.audio import Audio

""" Implemented Preprocessors:
- AudioReader - Read audio from path and return audio and label
- WavReader - Read wav file with librosa and return spectrogram and label
"""

class PostProcessors:
    """Postprocess the output of the model.

    Args:
        audio_name (str): Name of the audio file.
        output (dict): Output of the model with time
            durations and transcriptions.
    """
    
    # def __init__(self, audio_name: str, output: dict):
    #     self.audio_name = audio_name
    #     self.output = output

    @staticmethod
    def seconds_to_hhmmss(seconds: float) -> str:
        """Convert a duration from seconds to hh:mm:ss format.

        Args:
            seconds (float): Duration in seconds.

        Returns:
            str: Duration in hh:mm:ss format.
        """
        # Convert the seconds to an integer number of seconds
        seconds = int(seconds)
        # Use datetime to format the seconds into hh:mm:ss
        return str(datetime.timedelta(seconds=seconds))
    
    def remove_temp_directory(self, audio_name:str, audios: list[str]) -> None:
        """Delete the directory temp and its contents.

        Args:
            audio_name (str): Name of the audio file.
            audios (list): List of audio paths.
        """
        # Remove the directory temp and its contents based on the name of the audio file
        # delete the directory temp and its contents
        for wav_path in audios:
            os.remove(wav_path)

        os.system(f"rmdir data/input/temp/{audio_name}/")
        

    def generate_text_output(self, output: dict, audio_name: str) -> None:
        """Generate a text file with the output.

        Args:
            output (dict): Output of the model with time durations and transcriptions.
            audio_name (str): Name of the audio file.
        """
        # remove the extension from the audio name
        audio_name = audio_name.split(".")[0]

        # Create a text file with the output
        with open(f"data/output/{audio_name}.txt", "w") as file:
            # Write the output to the text file
            file.write(f"{'='*60}\n")
            file.write(f"Áudio: {audio_name}.wav\n")
            file.write(f"{'='*60}\n")
            file.write("Transcrição:\n")

            for line in output:
                file.write(f"{line['start']} -> {line['end']}: {line['text']}\n")
            
        print(f"Output saved to data/output/{audio_name}.txt")        

    
    def __call__(self, output: dict, audio_name: str, audios_path: list[str]) -> None:
        """Postprocess the output of the model.

        Args:
            output (dict): Output of the model with time durations and transcriptions.
            audio_name (str): Name of the audio file.
            audios_path (list): List of audio file paths.
        """
        # Generate a text output with the transcriptions
        self.generate_text_output(output, audio_name)

        # If there is more than one audio file, remove the temporary directory
        if len(audios_path) > 1:
            self.remove_temp_directory(audio_name.split(".")[0], audios_path)
       
     