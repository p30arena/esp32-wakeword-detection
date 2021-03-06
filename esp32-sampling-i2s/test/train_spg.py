import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from model_spg import model_path, get_model_train, tf_get_file_and_label, set_params
from model_spg import preprocess_dataset, get_spg_and_label_id


# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

data_dir = pathlib.Path('out/data-spg')
assert(data_dir.exists())

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('Commands:', commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*.bin')
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
waveform_ds = files_ds.map(tf_get_file_and_label, num_parallel_calls=AUTOTUNE)

set_params(commands, AUTOTUNE)

if __name__ == "__main__":
    spectrogram_ds = waveform_ds.map(
        get_spg_and_label_id, num_parallel_calls=AUTOTUNE)

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

    def lr_scheduler(epoch, lr):
        if epoch <= 25:
            return 0.001
        elif epoch <= 60:
            return 0.0001
        else:
            return 0.00001

    EPOCHS = 180

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            # tf.keras.callbacks.LearningRateScheduler(
            #     lr_scheduler, verbose=0),
            tf.keras.callbacks.EarlyStopping(verbose=1, patience=50),
        ],
    )

    model.save(model_path)

    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()
