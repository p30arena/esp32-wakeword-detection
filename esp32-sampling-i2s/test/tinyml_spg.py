import tensorflow as tf
import hexdump

from model_spg import get_spg_and_label_id, get_model_test
from train_spg import waveform_ds, AUTOTUNE

spectrogram_ds = waveform_ds.map(
    get_spg_and_label_id, num_parallel_calls=AUTOTUNE)


def representative_dataset():
    for input_value, output_value in spectrogram_ds.batch(1):
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
with open("out/model-spg/model.tflite", "wb") as file:
    file.write(tflite_model)
with open("out/model-spg/model.c", "wb") as file:
    file.write(str.encode(c_code))
