# create_profile.py

import pveagle
from pvrecorder import PvRecorder

ACCESS_KEY = "uk+w3OfU8V1PdTqYW92pp1KbpXVCQ6lybyoLmNFt0v7mteaVRthsAQ=="  # ← Remplace par ta clé Picovoice
PROFILE_PATH = "mon_profil.eagle"

def main():
    profiler = pveagle.create_profiler(access_key=ACCESS_KEY)
    recorder = PvRecorder(device_index=-1, frame_length=profiler.min_enroll_samples)

    try:
        recorder.start()
        print("🎤 Parle normalement pour créer ton profil vocal (10-20 sec)...")

        enroll_percentage = 0.0
        while enroll_percentage < 100.0:
            frame = recorder.read()
            enroll_percentage, feedback = profiler.enroll(frame)
            print(f"Progression : {enroll_percentage:.1f}% | Qualité : {feedback}")

        recorder.stop()

        with open(PROFILE_PATH, "wb") as f:
            f.write(profiler.export().to_bytes())
        print(f"✅ Profil sauvegardé : {PROFILE_PATH}")

    finally:
        profiler.delete()
        recorder.delete()

if __name__ == "__main__":
    main()
