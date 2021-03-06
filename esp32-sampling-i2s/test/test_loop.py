import numpy as np
import tensorflow as tf
from time import sleep

from commons import freq
import capture
from model import get_model_test, get_spectrogram

i = 0
closed = False
duration = 1  # seconds
model = get_model_test()
last_half = []
n_captured = 0
data = []


def infer(frame_data, is_mid=False):
    frame_data = np.array(frame_data, dtype=np.float32) / 32768
    spectrogram = get_spectrogram(frame_data)
    spectrogram = spectrogram[None, :]
    prediction = model.predict(spectrogram)
    sm = tf.nn.softmax(prediction[0])
    idx = np.argmax(sm)

    if idx == 1 and not is_mid:
        print("{0} nothing happening here".format(i))
    elif idx == 0:
        print('\nprobability: {0}'.format(sm[0]))
        print("I'm at your service!")
        print("abreman.ir")
        print('')
        sleep(3)
        return True
    return False


def on_connected():
    print("CONNECTED!")


def on_frame(frame_data: bytes, num_data: list):
    global n_captured, data, i, last_half
    n_captured += 1

    data += num_data

    if n_captured % 2 == 0:
        mid_ok = False
        if len(last_half) > 0:
            mid_ok = infer(last_half + data[:8000], is_mid=True)
        if not mid_ok:
            infer(data)
        last_half = data[8000:]
        i += 1
        data = []
    else:
        return


while capture.loop(on_connected, on_frame) == 1:
    print("restarting")
