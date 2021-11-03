import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import tensorflow as tf

from commons import freq

from model import get_model_test_tinyml, get_spectrogram

print("SPEAK!")

frame_data = sd.rec(1 * freq, samplerate=freq,
                    channels=1, dtype=np.int16)
sd.wait()
sd.play(frame_data, freq)
sd.wait()
frame_data = frame_data.flatten()
frame_data += 22000

spectogram = get_spectrogram(frame_data.astype(np.int8))
# spectogram = get_spectrogram(wf)
spectogram = spectogram[None, :]
model = get_model_test()
prediction = model.predict(spectogram)
print(prediction[0])
plt.bar(commands, tf.nn.softmax(prediction[0]))
plt.show()
