import socket
import wave
from commons import freq


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.settimeout(10000)
    s.connect(('192.168.1.89', 8840))
    with open('out.txt', 'w') as f:
        with wave.open('out.wav', 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # number of bytes
            w.setframerate(freq)
            samples = 0
            data = bytes()
            while(True):
                try:
                    chunk = s.recv(freq)
                    append_rest = False
                    n_appended = 0
                    n_appended = min(len(chunk), freq - len(data))
                    data += chunk[:n_appended]
                    if freq == len(data):
                        print('frame complete')
                        i = 0
                        while i < len(data):
                            n = data[i+1] << 8
                            n |= data[i]
                            f.write("{0}\n".format(n))
                            i += 2
                        w.writeframes(data)
                        data = bytes()
                        if n_appended < len(chunk):
                            data += chunk[n_appended:]
                except KeyboardInterrupt:
                    break
