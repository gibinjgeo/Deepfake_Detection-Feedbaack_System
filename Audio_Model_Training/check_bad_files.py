import torchaudio
import pathlib

root = pathlib.Path("Train/train/fake")
bad = []
for f in root.glob("*.mp3"):
    try:
        torchaudio.load(f)
    except Exception as e:
        bad.append((f, str(e)))

print("Bad files:", bad)
