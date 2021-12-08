import os
import pathlib

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow_model_optimization as tfmot

from commons import freq
from spg import decode_spg

commands = None
AUTOTUNE = None

model_path = pathlib.Path('out/model-spg')


def set_params(c, a):
    global commands, AUTOTUNE
    commands = c
    AUTOTUNE = a


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)

    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


def get_file_and_label(file_path):
    label = get_label(file_path)
    with open(file_path.numpy().decode('ascii'), 'rb') as f:
        return decode_spg(f.read()), label


def tf_get_file_and_label(file_path):
    a, b = tf.py_function(func=get_file_and_label, inp=[
                          file_path], Tout=[tf.float32, tf.string])
    a.set_shape((32, 32))
    b.set_shape(())
    return a, b


def get_spg_and_label_id(spectrogram, label):
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(tf_get_file_and_label,
                             num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spg_and_label_id,  num_parallel_calls=AUTOTUNE)
    return output_ds


def get_model_train(spectrogram_ds, input_shape, num_labels):
    model = None
    if model_path.exists():
        model = tf.keras.models.load_model(model_path)
    else:
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(8, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.5),
            layers.Conv2D(16, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(1),
        ])
        # model = tfmot.quantization.keras.quantize_model(model)

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
