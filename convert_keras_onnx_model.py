import tensorflow as tf
import tf2onnx
import onnx
from tensorflow import keras
from keras.models import load_model

from utils.configs import BaseModelConfigs
from utils.tensorflow.losses import CTCloss
from utils.tensorflow.metrics import CERMetric, WERMetric


configs = BaseModelConfigs.load("Models/sst/202410190151/configs.yaml")
vocab = configs.vocab

cer_metric = CERMetric(vocabulary=vocab)
wer_metric = WERMetric(vocabulary=vocab)


custom_objects = {
    'CTCloss': CTCloss(),
    'CERMetric': cer_metric,
    'WERMetric': wer_metric
}

# model_path = os.path.join(configs.model_path, 'model.keras')
model_path = 'Models/sst/202410190151/model.onnx'

model = load_model(model_path, custom_objects=custom_objects)

onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, 'model_b.onnx')


# If get the error: TypeError: __init__() got an unexpected keyword argument 'reduction'
# Change the line 6 from utils/tensorflow/losses.py for:
#     def __init__(self, name: str = "CTCloss", reduction=tf.keras.losses.Reduction.AUTO) -> None: