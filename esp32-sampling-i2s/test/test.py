from scipy.io.wavfile import write
import numpy as np
import matplotlib.pyplot as plt
from commons import freq

with open('out.txt', 'r') as f:
    data = np.array(f.readlines(), dtype=np.int16)
    write("example.wav", freq, data)
    plt.plot(list(range(len(data))), data)
    plt.show()
