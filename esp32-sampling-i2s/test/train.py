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

_90p = round(0.9 * num_samples)
_5p = round(0.05 * num_samples)

train_files = filenames[:_90p]
val_files = filenames[_90p: _90p + _5p]
test_files = filenames[-_5p:]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))


AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

set_params(commands, AUTOTUNE)

if __name__ == "__main__":
    # rows = 3
    # cols = 3
    # n = rows*cols
    # fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
    # for i, (audio, label) in enumerate(waveform_ds.take(n)):
    #     r = i // cols
    #     c = i % cols
    #     ax = axes[r][c]
    #     ax.plot(audio.numpy())
    #     ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
    #     label = label.numpy().decode('utf-8')
    #     ax.set_title(label)

    # plt.show()

    for waveform, label in waveform_ds.take(1):
        label = label.numpy().decode('utf-8')
        spectrogram = get_spectrogram(waveform)

    print('Label:', label)
    print('Waveform shape:', waveform.shape)
    print('Spectrogram shape:', spectrogram.shape)
    # print('Audio playback')
    # sd.play(waveform[:, 0], samplerate=16000,
    #         channels=1, dtype=np.int16)

    def plot_spectrogram(spectrogram, ax):
        # Convert to frequencies to log scale and transpose so that the time is
        # represented in the x-axis (columns). An epsilon is added to avoid log of zero.
        log_spec = np.log(spectrogram.T+np.finfo(float).eps)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = range(height)
        ax.pcolormesh(X, Y, log_spec)

    # fig, axes = plt.subplots(2, figsize=(12, 8))
    # timescale = np.arange(waveform.shape[0])
    # axes[0].plot(timescale, waveform.numpy())
    # axes[0].set_title('Waveform')
    # axes[0].set_xlim([0, 16000])
    # plot_spectrogram(spectrogram.numpy(), axes[1])
    # axes[1].set_title('Spectrogram')
    # plt.show()

    spectrogram_ds = waveform_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

    # rows = 3
    # cols = 3
    # n = rows*cols
    # fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    # for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
    #     r = i // cols
    #     c = i % cols
    #     ax = axes[r][c]
    #     plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
    #     ax.set_title(commands[label_id.numpy()])
    #     ax.axis('off')

    # plt.show()

    train_ds = spectrogram_ds
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)

    # batch_size = 64
    # batch_size = 32
    batch_size = 16
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape
    print('Input shape:', input_shape)
    num_labels = len(commands)

    model = get_model_train(spectrogram_ds, input_shape, num_labels)

    # test_loss, test_acc = model.evaluate(val_ds, verbose=2)

    # print('\nVal accuracy:', test_acc)

    # test_audio = []
    # test_labels = []

    # for audio, label in test_ds:
    #     test_audio.append(audio.numpy())
    #     test_labels.append(label.numpy())

    # test_audio = tf.convert_to_tensor(test_audio)
    # test_labels = tf.convert_to_tensor(test_labels)

    # test_loss, test_acc = model.evaluate(test_audio, test_labels, verbose=2)

    # print('\nTest accuracy:', test_acc)
    # exit()

    EPOCHS = 100
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=10),
    )

    model.save(model_path)

    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()
