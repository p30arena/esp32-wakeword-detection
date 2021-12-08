import sounddevice as sd
import numpy as np
from commons import get_fileno, freq, le2be
from capture import write_frame_wave, write_num_list

fileno = get_fileno('out/data/cmd-1')
duration = 1  # seconds
print("Recording Audio")
frame_data = sd.rec(duration * freq, samplerate=freq,
                    channels=1, dtype=np.int16)
sd.wait()
frame_data = frame_data.flatten()
# frame_data *= -1
# frame_data += 22000
# frame_data += (-1 * np.iinfo(np.int16).min) // 2
# num_data = []
# le2be(frame_data, num_data)
write_frame_wave("out/data/cmd-1/{0}.wav".format(fileno), frame_data)
write_num_list("out/data/cmd-1/{0}.txt".format(fileno), frame_data)
