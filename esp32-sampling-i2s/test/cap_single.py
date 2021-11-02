import os
import capture

filename = '0'

_files = list(map(lambda f: int(f.split('.')[0]), filter(
    lambda f: f.endswith('.wav'), os.listdir('out/cmd-1'))))
if len(_files):
    _files.sort(reverse=True)
    fileno = str(_files[0] + 1)

n_captured = 0
frames = bytes()
data = []


def on_connected():
    print("SPEAK!")


def on_frame(frame_data: bytes, num_data: list):
    global n_captured, frames, data
    n_captured += 1

    if n_captured != 1:
        frames += frame_data
        data += num_data

    if n_captured == 3:
        capture.close()
        capture.write_frame_wave("out/cmd-1/{0}.wav".format(filename), frames)
        capture.write_num_list("out/cmd-1/{0}.txt".format(filename), data)
    else:
        return


capture.loop(on_connected, on_frame)
