import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import tensorflow as tf

from commons import freq

from model import get_model_test_tinyml, get_spectrogram
from train import commands

print("SPEAK!")

frame_data = sd.rec(1 * freq, samplerate=freq,
                    channels=1, dtype=np.int16)
sd.wait()
# sd.play(frame_data, freq)
# sd.wait()
frame_data = frame_data.flatten()
frame_data += 22000

model = get_model_test_tinyml()
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()
input_shape = input_details[0]['shape']

spectrogram = get_spectrogram(frame_data.astype(np.float32) / 32768)
# spectrogram = tf.cast(spectrogram * 2147483648, tf.int8)
# spectrogram = tf.cast(spectrogram * 32768, tf.int8)
# spectrogram = tf.cast(spectrogram * 255, tf.int8)
spectrogram = tf.image.convert_image_dtype(spectrogram, dtype=tf.int8)
spectrogram = spectrogram[np.newaxis, ..., np.newaxis]
print(spectrogram)
model.set_tensor(input_details[0]['index'], spectrogram)
model.invoke()

output_data = model.get_tensor(output_details[0]['index'])
print(output_data)
# output_data = output_data[0].astype(np.float) / 255
# print(output_data)

# plt.bar(commands, tf.nn.softmax(output_data))
# plt.show()
