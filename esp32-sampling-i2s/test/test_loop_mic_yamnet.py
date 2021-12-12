import sounddevice as sd
import numpy as np
import tensorflow as tf
from time import sleep

from commons import freq

from yamnet_commons import model_path

i = 0
closed = False
duration = 1  # seconds
model = tf.saved_model.load(
    str(model_path.parent.joinpath('./model-combined')))
last_half = np.array([])


def infer(frame_data, is_mid=False):
    frame_data = np.clip(frame_data, -32768, 32768)
    prediction = model(frame_data.astype(np.float32) / 32768)
    sm = tf.nn.softmax(prediction)
    idx = np.argmax(sm)

    if idx == 0:
        # print('\nprobability: {0}'.format(sm[0]))
        print("I'm at your service!")
        print("abreman.ir")
        print('')
        sleep(0.5)
        return True
    elif not is_mid:
        print("{0} nothing happening here".format(i))

    return False


while not closed:
    try:
        frame_data = sd.rec(duration * freq, samplerate=freq,
                            channels=1, dtype=np.int16)
        sd.wait()
        frame_data = frame_data.flatten()
        # frame_data += 22000  # if dataset is recorded with 22k booster
        mid_ok = False
        if len(last_half) > 0:
            mid_ok = infer(last_half + frame_data[:8000], is_mid=True)
        if not mid_ok:
            infer(frame_data)
        last_half = frame_data[8000:]
        i += 1
    except KeyboardInterrupt:
        closed = True
