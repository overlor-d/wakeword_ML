import sounddevice as sd
import soundfile as sf
import os
from pathlib import Path

# ================= CONFIG =================

MIC_INDEX = 9              # A50 X hw:2,0
SAMPLERATE = 48000
CHANNELS = 1
DURATION = 1.2             # secondes (1–1.5 recommandé)

BASE_DIR = Path("wakeword_data/ovos")
POS_DIR = BASE_DIR / "positive"
NEG_DIR = BASE_DIR / "negative"

# ==========================================

POS_DIR.mkdir(parents=True, exist_ok=True)
NEG_DIR.mkdir(parents=True, exist_ok=True)


def next_index(folder: Path, prefix: str) -> int:
    existing = list(folder.glob(f"{prefix}_*.wav"))
    if not existing:
        return 1
    nums = [
        int(f.stem.split("_")[-1])
        for f in existing
        if f.stem.split("_")[-1].isdigit()
    ]
    return max(nums) + 1


def record_sample(folder: Path, prefix: str):
    idx = next_index(folder, prefix)
    filename = folder / f"{prefix}_{idx:03d}.wav"

    print(f"→ Enregistrement {filename.name} ({DURATION}s)")
    audio = sd.rec(
        int(DURATION * SAMPLERATE),
        samplerate=SAMPLERATE,
        channels=CHANNELS,
        device=MIC_INDEX,
        dtype="int16",
        blocking=True,
    )

    sf.write(filename, audio, SAMPLERATE)
    print("✓ Sauvegardé\n")


def main():
    print("Mode enregistrement wakeword")
    print("Entrée : enregistrer")
    print("p + Entrée : basculer POSITIF (OVOS)")
    print("n + Entrée : basculer NÉGATIF")
    print("q + Entrée : quitter\n")

    mode = "positive"

    while True:
        cmd = input(f"[mode={mode}] > ").strip().lower()

        if cmd == "q":
            break
        elif cmd == "p":
            mode = "positive"
            print("→ Mode POSITIF (dire 'OVOS')\n")
        elif cmd == "n":
            mode = "negative"
            print("→ Mode NÉGATIF (ne PAS dire OVOS)\n")
        else:
            if mode == "positive":
                record_sample(POS_DIR, "ovos")
            else:
                record_sample(NEG_DIR, "neg")


if __name__ == "__main__":
    main()
