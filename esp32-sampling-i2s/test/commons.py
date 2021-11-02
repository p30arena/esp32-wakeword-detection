import os

freq = 16000


def get_fileno(p: str):
    _files = list(map(lambda f: int(f.split('.')[0]), filter(
        lambda f: f.endswith('.wav'), os.listdir(p))))
    if len(_files):
        _files.sort(reverse=True)
        return _files[0] + 1
    else:
        return 0


def le2be(frame_data, num_data):
    i = 0
    while i < len(frame_data):
        n = frame_data[i+1] << 8
        n |= frame_data[i]
        num_data.append(n)
        i += 2
