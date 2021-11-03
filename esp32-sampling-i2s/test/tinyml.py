import tensorflow as tf
import hexdump

from model import get_spectrogram_and_label_id, get_model_test
from train import waveform_ds, AUTOTUNE

spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)


def representative_dataset():
    for input_value, output_value in spectrogram_ds.batch(1).take(100):
        yield [input_value]


model = get_model_test()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()
bytes = hexdump.dump(tflite_model).split(' ')
c_array = ', '.join(['0x%02x' % int(byte, 16) for byte in bytes])
c = 'const unsigned char model_data[] __attribute__((aligned(8))) = {%s};' % (
    c_array)
c += '\nconst int model_data_len = %d;' % (len(bytes))
c_code = c

# Save the model to disk
with open("out/model/model.tflite", "wb") as file:
    file.write(tflite_model)
with open("out/model/model.c", "wb") as file:
    file.write(str.encode(c_code))
