import typing
import numpy as np

from utils.configs import BaseModelConfigs
from utils.inference_model_utils import OnnxInferenceModel
from utils.postprocessors import PostProcessors
from utils.preprocessors import PreProcessors, WavEnhancer, WavReader, WavSplitter
from utils.text_utils import ctc_decoder, get_cer, get_wer

import os


class WavToTextModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, data: np.ndarray):
        data_pred = np.expand_dims(data, axis=0)

        preds = self.model.run(self.output_names, {self.input_names[0]: data_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    configs = BaseModelConfigs.load("Models/sst/202410260206/configs.yaml")
    model = WavToTextModel(model_path=configs.model_path, char_list=configs.vocab, force_cpu=False)

    audio_name = "interrogatório-e-sentença-penal-do-vulgo-loucura-condenado-por-três-homicídios.wav"

    pre_processors = PreProcessors()
    
    audios = pre_processors(audio_name)

    output= []
    final_duration = 0

    for wav_path in audios:
        
        spectrogram = WavReader.get_spectrogram(wav_path, frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length)

        duration = spectrogram.shape[0] * configs.frame_step / 22050
        
        padded_spectrogram = np.pad(spectrogram, ((0, configs.max_spectrogram_length - spectrogram.shape[0]),(0,0)), mode="constant", constant_values=0)

        text = model.predict(padded_spectrogram)
        print(f"{PostProcessors.seconds_to_hhmmss(final_duration)} -> {PostProcessors.seconds_to_hhmmss(final_duration+duration)}: {text}")
        output.append({
            "start": PostProcessors.seconds_to_hhmmss(final_duration),
            "end": PostProcessors.seconds_to_hhmmss(final_duration+duration),
            "text": text
        })
        final_duration += duration

    post_processor = PostProcessors()
    post_processor(output=output, audio_name=audio_name, audios_path=audios)
