import tensorflow as tf
import hexdump

from yamnet_commons import train_ds, model_path


def representative_dataset():
    for batch, _ in train_ds.take(1):
        for input_value in batch:
            yield [input_value]


model = tf.keras.models.load_model(
    model_path.parent.joinpath('./model-combined'))
model.summary()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_model = converter.convert()

# Save the model to disk
with open("out/model/model.tflite", "wb") as file:
    file.write(tflite_model)
