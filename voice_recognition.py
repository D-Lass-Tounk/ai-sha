import time
import os
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import sounddevice as sd
import scipy.io.wavfile as wavfile
import speech_recognition as sr

def record_audio(duration=2, fs=16000):
    print(f"Enregistrement audio {duration}s... Parlez maintenant.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return recording.squeeze()

def transcribe_audio(audio, fs=16000):
    # Sauvegarde temporaire
    tmp_filename = "temp_audio.wav"
    wavfile.write(tmp_filename, fs, audio)
    r = sr.Recognizer()
    with sr.AudioFile(tmp_filename) as source:
        audio_data = r.record(source)
    try:
        text = r.recognize_google(audio_data, language="fr-FR")  # Ou 'en-US' selon ta langue
        return text
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        print(f"Erreur transcription : {e}")
        return ""

def load_profiles(profiles_folder="profiles"):
    encoder = VoiceEncoder()
    profiles = {}
    for fn in os.listdir(profiles_folder):
        if fn.endswith(".wav"):
            path = os.path.join(profiles_folder, fn)
            wav = preprocess_wav(path)
            embed = encoder.embed_utterance(wav)
            name = os.path.splitext(fn)[0]
            profiles[name] = embed
            print(f"Profil chargé : {name}")
    return profiles

def recognize_speaker(profiles, audio, encoder, threshold=0.6):
    embedding = encoder.embed_utterance(audio)
    distances = {
        name: np.dot(embedding, embed) / (np.linalg.norm(embedding) * np.linalg.norm(embed))
        for name, embed in profiles.items()
    }
    if not distances:
        return None, 0
    best_match = max(distances, key=distances.get)
    score = distances[best_match]
    if score > threshold:
        return best_match, score
    else:
        return None, score

def run_voice_loop(change_mode_callback, authorized_profile="toi", threshold=0.6):
    profiles = load_profiles()
    encoder = VoiceEncoder()
    cooldown_seconds = 5
    last_recognized = None
    last_time = 0

    print("Reconnaissance vocale démarrée. Parlez...")

    while True:
        now = time.time()
        audio_raw = record_audio(duration=2)
        text = transcribe_audio(audio_raw)
        print(f"Transcription : '{text}'")

        if now - last_time > cooldown_seconds:
            try:
                # Reconnaissance par Resemblyzer
                wav_preprocessed = preprocess_wav(audio_raw)
                result, score = recognize_speaker(profiles, wav_preprocessed, encoder, threshold)
                if result and result != last_recognized:
                    print(f"Voix reconnue: {result} (score: {score:.3f})")
                    if result == authorized_profile:
                        print(f"[Vocal] Voix autorisée détectée : {result}. Changement de mode activé.")
                        change_mode_callback("visiteur")
                    else:
                        print(f"[Vocal] Voix non autorisée: {result}. Pas de changement de mode.")
                    last_recognized = result
                elif not result:
                    print(f"Voix inconnue ou score insuffisant ({score:.3f})")
                    last_recognized = None
            except Exception as e:
                print(f"Erreur reconnaissance vocale avancée : {e}")
            last_time = now

        time.sleep(0.1)

if __name__ == "__main__":
    def dummy_mode_change(mode):
        print(f"=== Mode demandé: {mode} ===")

    run_voice_loop(dummy_mode_change)
