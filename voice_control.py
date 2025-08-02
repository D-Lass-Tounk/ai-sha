import speech_recognition as sr
import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("J'écoute...")
        audio = r.listen(source, phrase_time_limit=5)
    try:
        text = r.recognize_sphinx(audio)  # 100% local, sinon r.recognize_google(audio)
        print(f"Reconnu: {text}")
        return text.lower()
    except sr.UnknownValueError:
        print("Pas compris.")
        return ""
    except Exception as e:
        print(f"Erreur reconnaissance vocale: {e}")
        return ""

def check_command(text, keywords):
    return all(k in text for k in keywords)

def vocal_mode_control(change_mode_callback):
    keywords = ["active", "mode", "visiteur"]
    while True:
        phrase = listen()
        if phrase and check_command(phrase, keywords):
            speak("Voulez-vous que j'active le mode visiteur ?")
            response = listen()
            if "oui" in response:
                speak("Mode visiteur activé.")
                change_mode_callback("visiteur")
            else:
                speak("Très bien.")
