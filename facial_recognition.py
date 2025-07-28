import cv2
import time
import os
import threading
import numpy as np
from PIL import Image
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog
from insightface.app import FaceAnalysis
from numpy import dot
from numpy.linalg import norm
import mediapipe as mp
import smtplib
from email.message import EmailMessage

# === Couleur des points (visage + main) ===
POINT_COLOR = (200, 255, 255)  # Bleu clair

# Fonction pour envoyer un e-mail
def send_email(subject, body, sender, receiver, password):
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        print("📧 E-mail envoyé avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi de l'e-mail : {e}")

# Bip multiplateforme
try:
    import winsound
    def play_beep():
        for _ in range(3):
            winsound.Beep(800, 120)
        time.sleep(0.3)
except ImportError:
    import simpleaudio as sa
    def play_beep():
        wave_obj = sa.WaveObject.from_wave_file("beep.wav")
        for _ in range(3):
            wave_obj.play()
            time.sleep(0.12)
        time.sleep(0.3)

# Variables globales
model = None
known_encodings = []
known_names = []
last_face_bbox = None
last_frame = None
frame_lock = threading.Lock()
annotated_frame = None
<<<<<<< HEAD
email_sent = False

# Mémoire tampon pour visages inconnus
recent_unknown_encodings = []
MAX_RECENT_UNKNOWN = 50
SIMILARITY_THRESHOLD = 0.38
=======
unknown_face_counts = {}
MAX_UNKNOWN_SAVES = 3
email_sent = False  # pour éviter d’envoyer plusieurs fois
>>>>>>> 80f8df7 (Premier commit - version initiale de AI-SHA)

# Cosine similarity
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# Charger les visages
def load_known_faces(folder="faces"):
    encodings, names = [], []
    print(f"\n🔄 Chargement des visages depuis : {folder}\n")
    os.makedirs(folder, exist_ok=True)
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, filename)
            print(f"🔎 Analyse du fichier : {filename}")
            try:
                img = np.array(Image.open(path).convert("RGB"))
                faces = model.get(img)
                if faces:
                    encodings.append(faces[0].embedding)
                    name = filename.split("_")[0]
                    names.append(name)
                    print(f"✅ Visage chargé : {name}")
                else:
                    print(f"❌ Aucun visage détecté dans : {filename}")
            except Exception as e:
                print(f"⚠️ Erreur sur {filename} : {e}")
    return encodings, names

# Enregistrement manuel
def save_face():
    global last_face_bbox, last_frame, known_encodings, known_names
    if last_face_bbox is None or last_frame is None:
        print("⛔ Aucun visage à sauvegarder.")
        return
    y1, x2, y2, x1 = last_face_bbox
    top = max(0, y1 - 20)
    bottom = min(last_frame.shape[0], y2 + 20)
    left = max(0, x1 - 20)
    right = min(last_frame.shape[1], x2 + 20)
    face_img = last_frame[top:bottom, left:right]

    root = tk.Tk()
    root.withdraw()
    name = simpledialog.askstring("Nom", "Entrez le nom du visage :")
    root.destroy()

    if not name:
        print("⛔ Aucun nom entré.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.jpg"
    path = os.path.join("faces", filename)
    cv2.imwrite(path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
    print(f"📂 Visage sauvegardé : {path}")

    faces = model.get(face_img)
    if faces:
        known_encodings.append(faces[0].embedding)
        known_names.append(name)
        print(f"🆕 {name} ajouté à la base.")
    else:
        print("⚠️ Échec de l'encodage → veuillez réessayer.")

# Thread de détection (visage + main)
def detection_thread_func():
<<<<<<< HEAD
    global annotated_frame, last_frame, last_face_bbox, email_sent, recent_unknown_encodings
=======
    global annotated_frame, last_frame, last_face_bbox, email_sent
>>>>>>> 80f8df7 (Premier commit - version initiale de AI-SHA)
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
    while True:
        with frame_lock:
            if last_frame is None:
                continue
            frame_rgb = last_frame.copy()
        result_frame = frame_rgb.copy()
        faces = model.get(frame_rgb)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            emb = face.embedding
            label = "Visage inconnu"
            similarity = 0

            if known_encodings:
                scores = [cosine_similarity(emb, ref) for ref in known_encodings]
                max_score = max(scores)
                best_match = np.argmax(scores)
<<<<<<< HEAD
                if max_score > SIMILARITY_THRESHOLD:
                    label = known_names[best_match]
                    similarity = max_score

=======
                if max_score > 0.38:
                    label = known_names[best_match]
                    similarity = max_score

            # Points du visage
>>>>>>> 80f8df7 (Premier commit - version initiale de AI-SHA)
            if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None:
                for (x, y, _) in face.landmark_3d_68.astype(int):
                    cv2.circle(result_frame, (x, y), 1, POINT_COLOR, -1)

<<<<<<< HEAD
=======
            # Texte
>>>>>>> 80f8df7 (Premier commit - version initiale de AI-SHA)
            text = f"{label} ({int(similarity * 100)}%)" if similarity else label
            cv2.putText(result_frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (64, 64, 64), 1)

            if label == "Visage inconnu":
<<<<<<< HEAD
                is_new_unknown = True
                for unk_emb in recent_unknown_encodings:
                    if cosine_similarity(emb, unk_emb) > SIMILARITY_THRESHOLD:
                        is_new_unknown = False
                        break

                if is_new_unknown:
                    play_beep()
                    last_face_bbox = (y1, x2, y2, x1)
                    face_img = frame_rgb[y1:y2, x1:x2]
                    os.makedirs("unknown", exist_ok=True)
                    filename = f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(os.path.join("unknown", filename), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    print("📸 Visage inconnu sauvegardé.")

                    recent_unknown_encodings.append(emb)
                    if len(recent_unknown_encodings) > MAX_RECENT_UNKNOWN:
                        recent_unknown_encodings.pop(0)

=======
                play_beep()
                last_face_bbox = (y1, x2, y2, x1)
                emb_key = tuple(np.round(emb[:10], 3))
                if unknown_face_counts.get(emb_key, 0) < MAX_UNKNOWN_SAVES:
                    os.makedirs("unknown", exist_ok=True)
                    filename = f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    face_img = frame_rgb[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join("unknown", filename), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                    unknown_face_counts[emb_key] = unknown_face_counts.get(emb_key, 0) + 1
                    print(f"📸 Visage inconnu sauvegardé ({unknown_face_counts[emb_key]}x)")

        # Détection des mains
>>>>>>> 80f8df7 (Premier commit - version initiale de AI-SHA)
        results = mp_hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = [(int(lm.x * frame_rgb.shape[1]), int(lm.y * frame_rgb.shape[0]))
                          for lm in hand_landmarks.landmark]
                for x, y in points:
                    cv2.circle(result_frame, (x, y), 2, POINT_COLOR, -1)
<<<<<<< HEAD
=======

                # Relier les points
>>>>>>> 80f8df7 (Premier commit - version initiale de AI-SHA)
                for connection in mp.solutions.hands.HAND_CONNECTIONS:
                    start, end = connection
                    x1, y1 = points[start]
                    x2, y2 = points[end]
                    cv2.line(result_frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

<<<<<<< HEAD
=======
                # Compter les doigts levés
>>>>>>> 80f8df7 (Premier commit - version initiale de AI-SHA)
                finger_ids = [4, 8, 12, 16, 20]
                fingers_up = sum([1 for fid in finger_ids if points[fid][1] < points[fid - 2][1]])
                cx, cy = points[0]
                cv2.putText(result_frame, str(fingers_up), (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, POINT_COLOR, 2)

<<<<<<< HEAD
=======
                # Envoi e-mail si deux doigts levés
>>>>>>> 80f8df7 (Premier commit - version initiale de AI-SHA)
                if fingers_up == 2 and not email_sent:
                    send_email(
                        subject="🚨 ALERTE - SIGNALE DE DANGER DÉTECTÉ.",
                        body="Un signal de danger a été repéré : un geste à deux doigts a été détecté devant la caméra.",
                        sender="lasstounk2017@gmail.com",
                        receiver="lasstounk2023@gmail.com",
                        password="pczxinliuvioazxj"
                    )
<<<<<<< HEAD
                    email_sent = True
=======
                    email_sent = True  # Évite de le renvoyer plusieurs fois
>>>>>>> 80f8df7 (Premier commit - version initiale de AI-SHA)

        with frame_lock:
            annotated_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)

# Main
def main():
    global model, known_encodings, known_names, last_frame
    model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    model.prepare(ctx_id=0)
    known_encodings[:], known_names[:] = load_known_faces()

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 30)
    if not cam.isOpened():
        print("❌ Impossible d'ouvrir la webcam.")
        return

    print("\n📷 Caméra ouverte. Appuyez sur :")
    print(" - 's' pour enregistrer un visage")
    print(" - 'r' pour recharger la base")
    print(" - 'Échap' pour quitter\n")

    threading.Thread(target=detection_thread_func, daemon=True).start()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("⚠️ Problème de lecture de la caméra.")
            break

        with frame_lock:
            last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
            display = annotated_frame.copy() if annotated_frame is not None else frame.copy()

        cv2.imshow("AI-SHA - Sécurité Visage & Mains", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            save_face()
            known_encodings[:], known_names[:] = load_known_faces()
        elif key == ord('r'):
            print("🔁 Rechargement de la base...")
            known_encodings[:], known_names[:] = load_known_faces()

    cam.release()
    cv2.destroyAllWindows()
    print("👋 Fin du programme.")

if __name__ == "__main__":
    main()
