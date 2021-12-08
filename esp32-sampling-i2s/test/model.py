import os
import pathlib

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow_model_optimization as tfmot

from commons import freq

commands = None
AUTOTUNE = None

model_path = pathlib.Path('out/model')


def set_params(c, a):
    global commands, AUTOTUNE
    commands = c
    AUTOTUNE = a


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)

    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram(waveform):
    # Padding for files with less than [freq] samples
    zero_padding = tf.zeros([freq] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    # default was 255 and 128
    spectrogram = tf.signal.stft(
        equal_length, frame_length=64, frame_step=512)

    spectrogram = tf.abs(spectrogram)

    # removed layers.Resizing in the model so adding resize here
    spectrogram = tf.image.resize(spectrogram[..., None], (32, 32))
    spectrogram = tf.squeeze(spectrogram, axis=2)

    # return tf.signal.mfccs_from_log_mel_spectrograms(spectrogram)

    return spectrogram


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label,
                             num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
    return output_ds


def get_model_train(spectrogram_ds, input_shape, num_labels):
    model = None
    if model_path.exists():
        model = tf.keras.models.load_model(model_path)
    else:
        # Instantiate the `tf.keras.layers.Normalization` layer.
        norm_layer = layers.Normalization()
        # Fit the state of the layer to the spectrograms
        # with `Normalization.adapt`.
        norm_layer.adapt(data=spectrogram_ds.map(
            map_func=lambda spec, label: spec))

        model = models.Sequential([
            layers.Input(shape=input_shape),
            # Normalize.
            norm_layer,
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1),
        ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    return model


def get_model_test():
    return tf.keras.models.load_model(model_path)


def get_model_test_tinyml() -> tf.lite.Interpreter:
    return tf.lite.Interpreter(str(model_path.joinpath('./model.tflite')))
