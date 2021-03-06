import numpy as np
import tensorflow as tf

from yamnet_commons import filenames, model_path, load_wav_for_map, AUTOTUNE, commands

files_ds = tf.data.Dataset.from_tensor_slices(filenames)
files_ds = files_ds.map(load_wav_for_map, num_parallel_calls=AUTOTUNE)
files_ds = files_ds.cache()

model = tf.saved_model.load(
    str(model_path.parent.joinpath('./model-combined')))


results = {}

for frame_data, lbl_idx in files_ds:
    lbl_idx = lbl_idx.numpy()
    prediction = model(frame_data)
    sm = tf.nn.softmax(prediction)
    idx = np.argmax(sm)

    if lbl_idx not in results:
        results[lbl_idx] = {
            'n_total': 0,
            'n_ok': 0
        }

    results[lbl_idx]['n_total'] += 1
    if idx == lbl_idx:
        results[lbl_idx]['n_ok'] += 1


print(results)

for lbl, v in results.items():
    print("{0} accuracy: {1}".format(commands[lbl], v['n_ok']/v['n_total']))
