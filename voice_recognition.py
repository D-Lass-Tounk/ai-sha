# voice_recognition.py

import os
import pveagle
from pvrecorder import PvRecorder

access_key = "uk+w3OfU8V1PdTqYW92pp1KbpXVCQ6lybyoLmNFt0v7mteaVRthsAQ=="
profiles_dir = "./profiles"

def main_voice():
    speaker_profiles = []
    profile_names = []

    for filename in os.listdir(profiles_dir):
        if filename.endswith(".eagle"):
            profile_path = os.path.join(profiles_dir, filename)
            with open(profile_path, "rb") as f:
                profile_bytes = f.read()
            profile = pveagle.EagleProfile.from_bytes(profile_bytes)
            speaker_profiles.append(profile)
            profile_names.append(os.path.splitext(filename)[0])

    try:
        eagle = pveagle.create_recognizer(
            access_key=access_key,
            speaker_profiles=speaker_profiles
        )
    except pveagle.EagleError as e:
        print(f"❌ Impossible de créer le moteur Eagle : {e}")
        return

    DEFAULT_DEVICE_INDEX = -1
    recorder = PvRecorder(
        device_index=DEFAULT_DEVICE_INDEX,
        frame_length=eagle.frame_length
    )

    THRESHOLD = 0.1
    print("🎧 Démarrage reconnaissance vocale...")

    recorder.start()
    try:
        while True:
            audio_frame = recorder.read()
            try:
                scores = eagle.process(audio_frame)
            except pveagle.EagleActivationError as e:
                print(f"⚠️ Erreur Eagle : {e}")
                break

            max_score = max(scores)
            max_index = scores.index(max_score)
            if max_score >= THRESHOLD:
                print(f"Profil reconnu : {profile_names[max_index]} (score: {max_score:.4f})")
            else:
                print("Aucun profil reconnu (voix inconnue ou silence)")
    except KeyboardInterrupt:
        print("🛑 Reconnaissance vocale arrêtée par l'utilisateur.")
    finally:
        recorder.stop()
        recorder.delete()
        eagle.delete()

if __name__ == "__main__":
    main_voice()
