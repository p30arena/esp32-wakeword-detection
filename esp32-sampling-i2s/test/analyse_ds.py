import subprocess
import struct
from glob import glob

import subprocess
files = glob('out/data/*/*.wav')

maxes = []
for _, f in enumerate(files):
    print("{0}/{1}".format(_, len(files)))
    buf = subprocess.run(
        '../../stft-conv/Release/stft-conv.exe 2 {0}'.format(f), capture_output=True).stdout
    result = []
    i = 0
    while i < len(buf):
        value = struct.unpack('d', buf[i:i+8])[0]
        result.append(int(abs(value)))
        i += 8
    result.sort()
    maxes.append(result[-1])

maxes.sort()
half_maxes = maxes[len(maxes) // -2:]

maxes_freq = {}

for m in half_maxes:
    if m in maxes_freq:
        maxes_freq[m] += 1
    else:
        maxes_freq[m] = 1

print(half_maxes)
print("\n\n\n")
print(maxes_freq)
print("\n")

maxes_clusters = {}

for k, v in maxes_freq.items():
    _100c = k // 10

    if _100c not in maxes_clusters:
        maxes_clusters[_100c] = 0
    maxes_clusters[_100c] += v

print(maxes_clusters)
