from pydub import AudioSegment
import os

def cortar_audios(caminho_arquivo):
    # Carregar o áudio
    audio = AudioSegment.from_file(caminho_arquivo, format="wav")
    
    # Definir a duração máxima em milissegundos (8 segundos)
    duracao_maxima = 8 * 1000
    
    # Encontrar períodos de silêncio (ajustar os parâmetros conforme necessário)
    silences = [0] + [i for i, sample in enumerate(audio) if sample.dBFS < -50] + [len(audio)]
    
    partes = []
    inicio = 0
    
    for i in range(1, len(silences)):
        if (silences[i] - inicio) > duracao_maxima:
            parte = audio[inicio:silences[i-1]]
            partes.append(parte)
            inicio = silences[i-1]
    
    if inicio < len(audio):
        partes.append(audio[inicio:])
    
    total = 0
    # Salvar partes
    for idx, parte in enumerate(partes):
        print(f"Segmento {idx+1} salvo com duração de {len(parte) / 1000:.2f} segundos")
        total += len(parte)
        parte.export(f'output_40/parte_{idx+1}.wav', format="wav")

    print("Total:", total / 1000, "segundos")

# # Substitua "seu_arquivo.mp3" pelo caminho do seu arquivo de áudio
# caminho_arquivo = "interrogatório-e-sentença-penal-do-vulgo-loucura-condenado-por-três-homicídios.wav"
# cortar_audios(caminho_arquivo)


os.makedirs(f"data/input/temp/teste", exist_ok=True)