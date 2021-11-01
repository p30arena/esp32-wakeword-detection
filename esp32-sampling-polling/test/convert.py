import serial
import wave

freq = 3200


def flatten(t):
    return [item for sublist in t for item in sublist]


s = serial.Serial('COM8', baudrate=115200, timeout=1000)

with open('out.txt', 'w') as f:
    with wave.open('out.wav', 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # number of bytes
        w.setframerate(freq)
        samples = 0
        while(True):
            try:
                data = s.read(3)
                if data[2] != 10:
                    while s.read(1)[0] != 10:
                        pass
                    continue
                data = data[:2]
                n = int.from_bytes(data, 'little')
                f.write("{0}\n".format(n))
                samples += 1
                if samples == freq:
                    samples = 0
                    print('frame complete')
                # w.writeframesraw(n.to_bytes(2, byteorder='big'))
                w.writeframesraw(data)
                # w.writeframes(data)
                # w.writeframes(n.to_bytes(2, byteorder='big'))
            except KeyboardInterrupt:
                break
