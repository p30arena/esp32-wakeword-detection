from scipy.io.wavfile import write
import numpy as np
import matplotlib.pyplot as plt

with open('out.txt', 'r') as f:
    data = np.array(f.readlines(), dtype=np.int16)
    write("example.wav", 16000, data)
    plt.plot(list(range(len(data))), data)
    plt.show()

# samplerate = 44100
# fs = 100
# t = np.linspace(0., 1., samplerate)
# amplitude = np.iinfo(np.int16).max
# data = amplitude * np.sin(2. * np.pi * fs * t)
# write("example.wav", samplerate, data.astype(np.int16))
# plt.plot(list(range(len(data))), data.astype(np.int16))
# plt.show()
