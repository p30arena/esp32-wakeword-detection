import socket
import wave
from commons import freq
from threading import Thread
from time import sleep
from typing import Callable

closed = False


def capture(events: list):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(3.0)
        s.connect(('192.168.1.89', 8840))
        events.insert(0, ("on_connected",))
        frame_data = bytes()
        while not closed:
            try:
                chunk = s.recv(freq)
                n_appended = 0
                n_appended = min(len(chunk), freq - len(frame_data))
                frame_data += chunk[:n_appended]
                if freq == len(frame_data):
                    i = 0
                    num_data = []
                    while i < len(frame_data):
                        n = frame_data[i+1] << 8
                        n |= frame_data[i]
                        num_data.append(n)
                        i += 2
                    events.insert(0, ("on_frame", frame_data, num_data))
                    frame_data = bytes()
                    if n_appended < len(chunk):
                        frame_data += chunk[n_appended:]
            except:
                close()


def begin(events: list) -> Thread:
    t = Thread(target=capture, args=(events,))
    t.start()
    return t


def close() -> None:
    global closed
    closed = True


def write_frame_wave(filename: str, frame: bytes):
    with wave.open(filename, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # number of bytes
        w.setframerate(freq)
        w.writeframes(frame)


def write_num_list(filename: str, lst: list):
    with open(filename, 'w') as f:
        f.write('\n'.join(map(lambda n: str(n), lst)))


def loop(on_connected: Callable, on_frame: Callable[[bytes, list], None]):
    events = []

    t = begin(events)

    try:
        while not closed:
            if len(events) == 0:
                sleep(0.1)
                continue
            else:
                ev, *args = events.pop()

                if ev == "on_connected":
                    on_connected()
                elif ev == "on_frame":
                    frame_data, num_data = args
                    on_frame(frame_data, num_data)
    except KeyboardInterrupt:
        close()
        print("exiting")
        t.join()
