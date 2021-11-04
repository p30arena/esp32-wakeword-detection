import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops

from commons import freq
from model import get_model_test_tinyml, get_spectrogram, decode_audio
from train import commands

print("SPEAK!")

# frame_data = sd.rec(1 * freq, samplerate=freq,
#                     channels=1, dtype=np.int16)
# sd.wait()
# sd.play(frame_data, freq)
# sd.wait()
# frame_data = frame_data.flatten()
# frame_data += 22000

model = get_model_test_tinyml()
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()
input_shape = input_details[0]['shape']

for i in range(80):
    audio_binary = tf.io.read_file("out/data/cmd-1/{0}.wav".format(i))
    decoded_audio = decode_audio(audio_binary)
    spectrogram = get_spectrogram(decoded_audio)
    # spectrogram = get_spectrogram(frame_data.astype(np.float32) / 32768)
    spectrogram = tf.cast(spectrogram * 128 - 128, tf.int8)
    spectrogram = spectrogram[np.newaxis, ..., np.newaxis]
    model.set_tensor(input_details[0]['index'], spectrogram)
    model.invoke()

    output_data = model.get_tensor(output_details[0]['index'])
    # print(tf.argmax(tf.nn.softmax(output_data[0])).numpy())
    idx = tf.argmax(tf.nn.softmax(
        output_data[0].astype(np.float) / 128)).numpy()
    print(idx)
    # output_data = output_data[0].astype(np.float) / 128
    # print(output_data)

    # plt.bar(commands, tf.nn.softmax(output_data))
    # plt.show()
