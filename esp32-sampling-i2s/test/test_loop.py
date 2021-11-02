import sounddevice as sd
import numpy as np
import tensorflow as tf
from time import sleep

from commons import freq

from model import get_model_test, get_spectrogram
from train import AUTOTUNE, commands

i = 0
closed = False
duration = 1  # seconds
model = get_model_test()

while not closed:
    try:
        frame_data = sd.rec(duration * freq, samplerate=freq,
                            channels=1, dtype=np.int16)
        sd.wait()
        frame_data = frame_data.flatten()
        frame_data += 22000

        spectogram = get_spectrogram(frame_data.astype(np.float32) / 32768)
        spectogram = spectogram[None, :]
        prediction = model.predict(spectogram)
        sm = tf.nn.softmax(prediction[0])
        idx = np.argmax(sm)

        if idx == 1:
            print("{0} nothing happening here".format(i))
        else:
            print('\nprobability: {0}'.format(sm[0]))
            print("I'm at your service!")
            print("abreman.ir")
            print('')
            sleep(3)
        i += 1
    except KeyboardInterrupt:
        closed = True
