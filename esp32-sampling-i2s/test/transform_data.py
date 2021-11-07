import subprocess
import glob

files = list(map(lambda s: s.replace('\\', '/'),
                 glob.glob('out/data/*/*.wav')))
for f in files:
    out = f.replace('data', 'data-spg').replace('wav', 'bin')
    subprocess.run(
        '../../stft-conv/Release/stft-conv.exe 1 {0} {1}'.format(f, out)).returncode
