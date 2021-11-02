import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import tensorflow as tf

from model import get_model_test, get_spectrogram
from train import commands

print("SPEAK!")

frame_data = sd.rec(1 * 16000, samplerate=16000,
                    channels=1, dtype=np.int16)
sd.wait()
sd.play(frame_data, 16000)
sd.wait()
frame_data = frame_data.flatten()
frame_data += 22000
spectogram = get_spectrogram(frame_data)

spectogram = spectogram[None, :]
model = get_model_test()
prediction = model.predict(spectogram)

plt.bar(commands, tf.nn.softmax(prediction[0]))
plt.show()
