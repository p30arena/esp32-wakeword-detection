import numpy as np
import tensorflow as tf
from model_spg import get_model_test_tinyml
from spg import decode_spg

model = get_model_test_tinyml()
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()
input_shape = input_details[0]['shape']

n_total = 0
n_ok = 0

for i in range(111):
    binary = open("out/data-spg/cmd-1/{0}.bin".format(i), 'rb').read()
    spectrogram = decode_spg(binary)
    # spectrogram = get_spectrogram(frame_data.astype(np.float32) / 32768)
    spectrogram = tf.cast(spectrogram * 255 - 128, tf.int8)
    spectrogram = spectrogram[np.newaxis, ..., np.newaxis]
    model.set_tensor(input_details[0]['index'], spectrogram)
    model.invoke()

    output_data = model.get_tensor(output_details[0]['index'])
    # print(tf.argmax(tf.nn.softmax(output_data[0])).numpy())
    idx = tf.argmax(tf.nn.softmax(
        output_data[0].astype(np.float) / 128)).numpy()
    print(idx)

    n_total += 1
    if idx == 0:
        n_ok += 1
    # output_data = output_data[0].astype(np.float) / 128
    # print(output_data)

    # plt.bar(commands, tf.nn.softmax(output_data))
    # plt.show()

print("accuracy: {0}".format(n_ok/n_total))
