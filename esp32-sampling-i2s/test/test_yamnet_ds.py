import numpy as np
import tensorflow as tf
from model_spg import get_model_test_tinyml
from spg import decode_spg
from glob import glob

from model import model_path
from yamnet_commons import yamnet_model, load_wav_16k_mono

model = tf.keras.models.load_model(model_path)


def compute(path, actual, lbl):
    n_total = 0
    n_ok = 0

    for f in glob(path):
        frame_data = load_wav_16k_mono(f)
        scores, embeddings, spectrogram = yamnet_model(frame_data)

        prediction = model.predict(embeddings)
        sm = tf.nn.softmax(prediction[0])
        idx = np.argmax(sm)

        n_total += 1
        if idx == actual:
            n_ok += 1
        # else:
        #     print(f)
        #     print(softmnax.numpy())

    print("{0} accuracy: {1}".format(lbl, n_ok/n_total))


compute("out/data/cmd-1/*.wav", 0, "CMD")
compute("out/data/other-1/*.wav", 1, "OTHER")
