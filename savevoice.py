import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import sounddevice as sd
import scipy.io.wavfile as wavfile
import time
import os

def record_profile(filename, duration=3, fs=16000):
    print(f"Enregistrement du profil vocal : {filename} (parle maintenant)")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavfile.write(filename, fs, recording)
    print(f"Profil enregistré dans {filename}")

def load_profiles(filenames):
    encoder = VoiceEncoder()
    profiles = {}
    for fn in filenames:
        wav = preprocess_wav(fn)
        embed = encoder.embed_utterance(wav)
        name = os.path.splitext(os.path.basename(fn))[0]
        profiles[name] = embed
        print(f"Profil chargé : {name}")
    return profiles

def recognize_speaker(profiles, threshold=0.6, fs=16000, duration=3):
    encoder = VoiceEncoder()
    print("Parlez maintenant...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav = recording.squeeze()

    embedding = encoder.embed_utterance(wav)

    distances = {name: np.dot(embedding, embed) / (np.linalg.norm(embedding) * np.linalg.norm(embed)) for name, embed in profiles.items()}
    best_match = max(distances, key=distances.get)
    score = distances[best_match]

    print(f"Meilleure correspondance: {best_match} (score: {score:.3f})")
    if score > threshold:
        return best_match
    else:
        return None

if __name__ == "__main__":
    profiles_folder = "profiles"
    os.makedirs(profiles_folder, exist_ok=True)

    # Pour créer un profil, décommente et exécute une fois à la fois :
    record_profile(os.path.join(profiles_folder, "toi.wav"))
    # record_profile(os.path.join(profiles_folder, "autre.wav"))

    profils = load_profiles([os.path.join(profiles_folder, f) for f in os.listdir(profiles_folder) if f.endswith(".wav")])

    last_recognized = None
    last_time = 0
    cooldown_seconds = 5

    while True:
        now = time.time()
        if now - last_time > cooldown_seconds:
            result = recognize_speaker(profils)
            if result and result != last_recognized:
                print(f"Voix reconnue: {result}")
                last_recognized = result
            elif not result:
                print("Voix inconnue")
                last_recognized = None
            last_time = now
        time.sleep(0.1)
