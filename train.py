import tensorflow as tf

try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

import os
import pandas as pd
from tqdm import tqdm

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from model import train_model
from configs import ModelConfigs
from utils.tensorflow.callbacks import Model2onnx, TrainLogger
from utils.tensorflow.data_provider import DataProvider
from utils.preprocessors import WavReader
from utils.tensorflow.losses import CTCloss
from utils.tensorflow.metrics import CERMetric, WERMetric
from utils.transformes import LabelIndexer, LabelPadding, SpectrogramPadding


dataset_path = "Dataset/cv-corpus-17.0/"
wavs_path = dataset_path + "/wavs/"

# Read metadata file and parse it
dataset_info = pd.read_csv(dataset_path + 'dataset_info.csv', sep=';')
# structure the dataset where each row is a list of [wav_file_path, sound transcription]
dataset = [[f"Dataset/cv-corpus-17.0/wavs/{file}", label.lower()] for file, label in dataset_info.values.tolist()]

# Create a ModelConfigs object to store model configurations
configs = ModelConfigs()

max_text_length, max_spectrogram_length = 0, 0

# Get the maximum text and spectrogram lengths
for file_path, label in tqdm(dataset, desc="Calculating max lengths", total=len(dataset)):
    spectrogram = WavReader.get_spectrogram(
        file_path,
        frame_length=configs.frame_length,
        frame_step=configs.frame_step,
        fft_length=configs.fft_length,
    )
                
    valid_label = [c for c in label if c in configs.vocab]
    max_text_length = max(max_text_length, len(valid_label))
    max_spectrogram_length = max(max_spectrogram_length, spectrogram.shape[0])
    configs.input_shape = [max_spectrogram_length, spectrogram.shape[1]]

configs.max_spectrogram_length = max_spectrogram_length
configs.max_text_length = max_text_length
configs.save()

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        WavReader(
            frame_length=configs.frame_length,
            frame_step=configs.frame_step,
            fft_length=configs.fft_length,
        ),
    ],
    transformers=[
        SpectrogramPadding(
            max_spectrogram_length=configs.max_spectrogram_length, padding_value=0
        ),
        LabelIndexer(configs.vocab),
        LabelPadding(
            max_word_length=configs.max_text_length, padding_value=len(configs.vocab)
        ),
    ],
)

# Split the dataset into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split = 0.8)  

# Creating TensorFlow model architecture
model = train_model(
    input_dim = configs.input_shape,
    output_dim = len(configs.vocab),
    dropout=0.5
)

# Compile the model and print summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate), 
    loss=CTCloss(), 
    metrics=[
        CERMetric(vocabulary=configs.vocab),
        WERMetric(vocabulary=configs.vocab)
        ],
    run_eagerly=False
)
model.summary(line_length=110)

# Define callbacks
earlystopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1, mode="min")
checkpoint = ModelCheckpoint(
    f"{configs.model_path}/model.h5",
    monitor="val_CER",
    verbose=1,
    save_best_only=True,
    mode="min",
)
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(
    monitor="val_CER", factor=0.8, min_delta=1e-10, patience=5, verbose=1, mode="auto"
)
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
    workers=configs.train_workers
)

# Save training and validation datasets as csv files
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))
