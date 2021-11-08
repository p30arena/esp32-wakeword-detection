import subprocess
import struct
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm


def decode_spg(buf):
    result = []
    i = 0
    while i < 8*1024:
        value = struct.unpack('d', buf[i:i+8])[0]
        result.append(value)
        i += 8
    result = np.reshape(np.abs(np.array(result)), (32, 32))
    return tf.convert_to_tensor(result, dtype=tf.float32)


def get_spectrogram(f):
    if isinstance(f, tf.Tensor):
        f = f.numpy().decode('ascii')
    buf = subprocess.run(
        '../../stft-conv/Release/stft-conv.exe 0 {0}'.format(f), capture_output=True).stdout
    return decode_spg(buf)


def get_mfcc(f):
    if isinstance(f, tf.Tensor):
        f = f.numpy().decode('ascii')
    buf = subprocess.run(
        '../../mfcc-conv/Release/mfcc-conv.exe 0 {0}'.format(f), capture_output=True).stdout
    result = []
    i = 0
    while i < 8*1274:
        value = struct.unpack('d', buf[i:i+8])[0]
        result.append(value)
        i += 8
    return tf.abs(tf.reshape(tf.convert_to_tensor(np.array(result), dtype=tf.float32), (26, 49)))


if __name__ == "__main__":
    def plot_spectrogram(spectrogram, ax):
        log_spec = np.log(spectrogram.T+np.finfo(float).eps)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = range(height)
        ax.pcolormesh(X, Y, log_spec)

    def plot_mfcc(mfcc, ax):
        cax = ax.imshow(mfcc, interpolation='nearest',
                        cmap=cm.coolwarm, origin='lower')

    rows = 3
    cols = 3
    n = rows*cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    idx = 0
    for i in range(9, 18):
        r = idx // cols
        c = idx % cols
        idx += 1
        ax = axes[r][c]
        spectrogram = get_spectrogram(
            "out/data/cmd-1/{0}.wav".format(i) if i % 2 == 0 else "out/data/other-1/{0}.wav".format(i))
        plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
        ax.set_title("cmd" if i % 2 == 0 else "other")
        ax.axis('off')

    plt.show()
