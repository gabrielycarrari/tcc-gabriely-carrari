from transformers import pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.text_utils import get_cer, get_wer

# Carregar o pipeline de reconhecimento de fala
pipe = pipeline("automatic-speech-recognition", model="lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2")

# Carregar o dataset de validação
df = pd.read_csv("Dataset/validation/data_validation_info.csv").values.tolist()

# Listas para acumular os CERs e WERs
accum_cer, accum_wer = [], []

# Iterar sobre o dataset e realizar a transcrição
for wav_path, true_label in tqdm(df):
    wav_path = f'Dataset/validation/wavs/{wav_path}'

    # Usar o pipeline para transcrever o áudio
    transcription = pipe(wav_path)
    predicted_text = transcription["text"] +'.'

    # Calcular o CER e WER
    cer = get_cer(predicted_text.lower(), true_label.lower())
    wer = get_wer(predicted_text.lower(), true_label.lower())

    accum_cer.append(cer)
    accum_wer.append(wer)


# Exibir as métricas médias
print(f"Average CER: {np.average(accum_cer)}")
print(f"Average WER: {np.average(accum_wer)}")