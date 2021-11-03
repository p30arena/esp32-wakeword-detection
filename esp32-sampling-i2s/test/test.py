import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import tensorflow as tf

from commons import freq

from model import get_model_test, get_spectrogram, preprocess_dataset, set_params, decode_audio
from train import AUTOTUNE, commands
from capture import write_frame_wave

print("SPEAK!")

frame_data = sd.rec(1 * freq, samplerate=freq,
                    channels=1, dtype=np.int16)
sd.wait()
sd.play(frame_data, freq)
sd.wait()
frame_data = frame_data.flatten()
frame_data += 22000


# write_frame_wave("out/test.wav", frame_data)
# model = get_model_test()
# sample_ds = preprocess_dataset(["out/test.wav"])
# print(sample_ds)
# for spectrogram, label in sample_ds.batch(1):
#     prediction = model(spectrogram)
#     plt.bar(commands, tf.nn.softmax(prediction[0]))
#     plt.title(f'Predictions for "{commands[label[0]]}"')
#     plt.show()

# audio_binary = tf.io.read_file("out/test.wav")
# wf = decode_audio(audio_binary)

spectrogram = get_spectrogram(frame_data.astype(np.float32) / 32768)
# spectrogram = get_spectrogram(wf)
spectrogram = spectrogram[None, :]
model = get_model_test()
prediction = model.predict(spectrogram)
print(prediction[0])
plt.bar(commands, tf.nn.softmax(prediction[0]))
plt.show()
