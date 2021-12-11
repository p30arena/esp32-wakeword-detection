import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

from model import model_path, get_label, get_waveform_and_label, set_params
from model import preprocess_dataset, get_spectrogram_and_label_id, get_spectrogram

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)


@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def load_wav_for_map(file_path):
    label = get_label(file_path)
    label_id = tf.argmax(label == commands)
    return load_wav_16k_mono(file_path), label_id


# applies the embedding extraction model to a wav data
def extract_embedding(wav_data, label):
    ''' run YAMNet to extract embedding from the wav data '''
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    num_embeddings = tf.shape(embeddings)[0]
    return (embeddings, tf.repeat(label, num_embeddings))


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


AUTOTUNE = tf.data.AUTOTUNE
set_params(commands, AUTOTUNE)

files_ds = tf.data.Dataset.from_tensor_slices(filenames)
files_ds = files_ds.map(load_wav_for_map)

# extract embedding
files_ds = files_ds.map(extract_embedding).unbatch()
files_ds = files_ds.cache()

_90p = round(0.9 * num_samples)
_5p = round(0.05 * num_samples)

train_ds = files_ds.take(_90p)
val_ds = files_ds.skip(_90p)
test_ds = val_ds.skip(_5p)
val_ds = test_ds.take(_5p)

train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                          name='input_embedding'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2)
], name='my_model')

model.summary()

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer="adam",
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=3,
                                            restore_best_weights=True)
