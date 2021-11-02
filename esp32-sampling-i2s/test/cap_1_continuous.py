import os
import capture

fileno = 0

_files = list(map(lambda f: int(f.split('.')[0]), filter(
    lambda f: f.endswith('.wav'), os.listdir('out/other-1'))))
if len(_files):
    _files.sort(reverse=True)
    fileno = _files[0] + 1

n_captured = 0
frames = bytes()
data = []


def on_connected():
    print("CONNECTED!")


def on_frame(frame_data: bytes, num_data: list):
    global n_captured, frames, data, fileno
    n_captured += 1

    frames += frame_data
    data += num_data

    if n_captured % 2 == 0:
        capture.write_frame_wave(
            "out/other-1/{0}.wav".format(fileno), frames)
        capture.write_num_list("out/other-1/{0}.txt".format(fileno), data)
        print("SAVED {0}".format(fileno))
        frames = bytes()
        data = []
        fileno += 1
    else:
        return


capture.loop(on_connected, on_frame)
