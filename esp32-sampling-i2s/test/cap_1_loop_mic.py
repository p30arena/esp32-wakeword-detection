import sounddevice as sd
import numpy as np
from commons import get_fileno, freq, le2be
from capture import close, write_frame_wave, write_num_list

closed = False
fileno = get_fileno('out/other-1')
duration = 1  # seconds

while not closed:
    try:
        frame_data = sd.rec(duration * freq, samplerate=freq,
                            channels=1, dtype=np.int16)
        sd.wait()
        frame_data = frame_data.flatten()
        frame_data += 22000
        write_frame_wave("out/other-1/{0}.wav".format(fileno), frame_data)
        write_num_list("out/other-1/{0}.txt".format(fileno), frame_data)
        print("SAVED {0}".format(fileno))
        fileno += 1
    except KeyboardInterrupt:
        closed = True
