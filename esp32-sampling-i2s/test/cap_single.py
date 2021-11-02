import capture
from commons import get_fileno

fileno = get_fileno('out/data/cmd-1')
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
        capture.write_frame_wave(
            "out/data/cmd-1/{0}.wav".format(fileno), frames)
        capture.write_num_list("out/data/cmd-1/{0}.txt".format(fileno), data)
    else:
        return


capture.loop(on_connected, on_frame)
