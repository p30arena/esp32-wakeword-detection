import numpy as np
import tensorflow as tf
from glob import glob

from model import get_model_test_tinyml, get_spectrogram, decode_audio

model = get_model_test_tinyml()
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()
input_shape = input_details[0]['shape']


def compute(path, actual, lbl):
    n_total = 0
    n_ok = 0

    for f in glob(path):
        audio_binary = tf.io.read_file(f)
        decoded_audio = decode_audio(audio_binary)
        spectrogram = get_spectrogram(decoded_audio)
        spectrogram = tf.cast(spectrogram * 255 - 128, tf.int8)
        spectrogram = spectrogram[np.newaxis, ..., np.newaxis]
        model.set_tensor(input_details[0]['index'], spectrogram)
        model.invoke()

        output_data = model.get_tensor(output_details[0]['index'])
        idx = tf.where(tf.nn.sigmoid(
            output_data[0].astype(np.float) / 128) < 0.5, 0, 1).numpy()[0]
        # softmnax = tf.nn.softmax(
        #     output_data[0].astype(np.float) / 128)
        # idx = tf.argmax(softmnax).numpy()

        n_total += 1
        if idx == actual:
            n_ok += 1
        # else:
        #     print(f)
        #     print(softmnax.numpy())

    print("{0} accuracy: {1}".format(lbl, n_ok/n_total))


compute("out/data/cmd-1/*.wav", 0, "CMD")
compute("out/data/other-1/*.wav", 1, "OTHER")
