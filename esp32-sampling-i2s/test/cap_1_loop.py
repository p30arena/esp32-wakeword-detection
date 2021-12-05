import capture
from commons import get_fileno

fileno = get_fileno('out/data/other-1')
frames = bytes()
data = []


def on_connected():
    print("CONNECTED!")


def on_frame(frame_data: bytes, num_data: list):
    global frames, data, fileno

    frames += frame_data
    data += num_data

    if len(frames) == 32000:
        capture.write_frame_wave(
            "out/data/other-1/{0}.wav".format(fileno), frames)
        capture.write_num_list("out/data/other-1/{0}.txt".format(fileno), data)
        print("SAVED {0}".format(fileno))
        frames = bytes()
        data = []
        fileno += 1
    else:
        return


while capture.loop(on_connected, on_frame) == 1:
    print("restarting")

print("exiting")
