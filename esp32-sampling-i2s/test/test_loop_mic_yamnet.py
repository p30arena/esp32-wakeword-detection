import sounddevice as sd
import numpy as np
import tensorflow as tf
from time import sleep

from commons import freq

from model import model_path
from yamnet_commons import yamnet_model

i = 0
closed = False
duration = 1  # seconds
model = tf.keras.models.load_model(model_path)
last_half = []


def infer(frame_data, is_mid=False):
    frame_data = np.clip(frame_data, -32768, 32768)
    scores, embeddings, spectrogram = yamnet_model(
        frame_data.astype(np.float32) / 32768)
    # spectrogram = get_spectrogram(frame_data.astype(np.float32) / 32768)
    # spectrogram = np.reshape(spectrogram, (32, 32, 1))[None, :]
    prediction = model.predict(embeddings)
    sm = tf.nn.softmax(prediction[0])
    idx = np.argmax(sm)
    # idx = tf.where(tf.nn.sigmoid(
    #     prediction[0].astype(np.float) / 128) < 0.5, 0, 1).numpy()[0]

    if idx == 1 and not is_mid:
        print("{0} nothing happening here".format(i))
    elif idx == 0:
        # print('\nprobability: {0}'.format(sm[0]))
        print("I'm at your service!")
        print("abreman.ir")
        print('')
        # sleep(3)
        return True
    return False


while not closed:
    try:
        frame_data = sd.rec(duration * freq, samplerate=freq,
                            channels=1, dtype=np.int16)
        sd.wait()
        frame_data = frame_data.flatten()
        frame_data += 22000  # this dataset is recorded with 22k booster
        mid_ok = False
        if len(last_half) > 0:
            mid_ok = infer(last_half + frame_data[:8000], is_mid=True)
        if not mid_ok:
            infer(frame_data)
        last_half = frame_data[8000:]
        i += 1
    except KeyboardInterrupt:
        closed = True
