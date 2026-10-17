"""Generate assets/sounds/whistle.wav: three short rescue-whistle blasts (international distress signal), then a pause.

Run from mobile/: python tool/make_whistle.py   (no dependencies; the app loops the file)
"""
import math
import struct
import wave
from pathlib import Path

RATE = 44100
OUT = Path(__file__).resolve().parent.parent / "assets" / "sounds" / "whistle.wav"


def blast(seconds: float) -> list[float]:
    samples = []
    n = int(RATE * seconds)
    for i in range(n):
        t = i / RATE
        # pea-whistle: ~3.1 kHz carrier with a fast warble and an overtone
        f = 3150 + 120 * math.sin(2 * math.pi * 28 * t)
        value = 0.8 * math.sin(2 * math.pi * f * t) + 0.2 * math.sin(2 * math.pi * 2 * f * t)
        envelope = min(1.0, i / (0.01 * RATE), (n - i) / (0.01 * RATE))  # 10 ms fades, no clicks
        samples.append(value * envelope)
    return samples


def silence(seconds: float) -> list[float]:
    return [0.0] * int(RATE * seconds)


samples = blast(0.4) + silence(0.2) + blast(0.4) + silence(0.2) + blast(0.4) + silence(0.9)
OUT.parent.mkdir(parents=True, exist_ok=True)
with wave.open(str(OUT), "wb") as file:
    file.setnchannels(1)
    file.setsampwidth(2)
    file.setframerate(RATE)
    file.writeframes(b"".join(struct.pack("<h", int(max(-1, min(1, s)) * 32000)) for s in samples))
print(OUT, len(samples) / RATE, "s")
