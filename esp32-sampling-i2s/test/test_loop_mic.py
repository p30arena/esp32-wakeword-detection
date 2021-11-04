import sounddevice as sd
import numpy as np
import tensorflow as tf
from time import sleep

from commons import freq

from model import get_model_test, get_spectrogram

i = 0
closed = False
duration = 1  # seconds
model = get_model_test()
last_half = []


def infer(frame_data, is_mid=False):
    spectrogram = get_spectrogram(frame_data.astype(np.float32) / 32768)
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


while not closed:
    try:
        frame_data = sd.rec(duration * freq, samplerate=freq,
                            channels=1, dtype=np.int16)
        sd.wait()
        frame_data = frame_data.flatten()
        frame_data += 22000
        mid_ok = False
        if len(last_half) > 0:
            mid_ok = infer(last_half + frame_data[:8000], is_mid=True)
        if not mid_ok:
            infer(frame_data)
        last_half = frame_data[8000:]
        i += 1
    except KeyboardInterrupt:
        closed = True
