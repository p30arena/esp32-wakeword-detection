import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from model import model_path, get_model_train, get_waveform_and_label, set_params
from model import preprocess_dataset, get_spectrogram_and_label_id, get_spectrogram


# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

data_dir = pathlib.Path('out/data')
assert(data_dir.exists())

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('Commands:', commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*.wav')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Number of examples per label:',
      len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
print('Example file tensor:', filenames[0])

_80p = round(0.8 * num_samples)
_10p = round(0.1 * num_samples)

train_files = filenames[:_80p]
val_files = filenames[_80p: _80p + _10p]
test_files = filenames[-_10p:]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))


AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

set_params(commands, AUTOTUNE)

if __name__ == "__main__":
    def plot_spectrogram(spectrogram, ax):
        # Convert to frequencies to log scale and transpose so that the time is
        # represented in the x-axis (columns). An epsilon is added to avoid log of zero.
        log_spec = np.log(spectrogram.T+np.finfo(float).eps)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = range(height)
        ax.pcolormesh(X, Y, log_spec)

    spectrogram_ds = waveform_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

    rows = 3
    cols = 3
    n = rows*cols
    n_lab = [0, 0]
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for _, (spectrogram, label_id) in enumerate(spectrogram_ds):
        i = n_lab[0] + n_lab[1]
        if i == n:
            break
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
        lid = label_id.numpy()
        if n_lab[lid] == 5:
            continue
        n_lab[lid] += 1
        ax.set_title(commands[lid])
        ax.axis('off')

    plt.show()
