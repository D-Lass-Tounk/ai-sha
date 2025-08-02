import os
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import simpledialog
import cv2
from datetime import datetime

from utils import cosine_similarity

known_encodings = []
known_names = []
last_face_bbox = None
last_frame = None
SIMILARITY_THRESHOLD = 0.38

# Remplacer par une liste de maîtres
MODE_MASTER_USERS = ["LASS", "HAOUA"]  # ajoute ici tous les maîtres autorisés

def load_known_faces(model, folder="faces"):
    encodings, names = [], []
    os.makedirs(folder, exist_ok=True)
    print(f"\nChargement des visages depuis : {folder}\n")
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, filename)
            print(f"Analyse du fichier : {filename}")
            try:
                img = np.array(Image.open(path).convert("RGB"))
                faces = model.get(img)
                if faces:
                    encodings.append(faces[0].embedding)
                    name = filename.split("_")[0]
                    names.append(name)
                    print(f"Visage chargé : {name}")
                else:
                    print(f"Aucun visage détecté dans : {filename}")
            except Exception as e:
                print(f"Erreur sur {filename} : {e}")
    return encodings, names

def ask_name():
    root = tk.Tk()
    root.withdraw()
    name = simpledialog.askstring("Nom", "Entrez le nom du visage :")
    root.destroy()
    return name

def add_face_to_known(model, known_encodings, known_names, image_path, name):
    try:
        img = np.array(Image.open(image_path).convert("RGB"))
        faces = model.get(img)
        if faces:
            known_encodings.append(faces[0].embedding)
            known_names.append(name)
            print(f"{name} ajouté à la base.")
            return True
        else:
            print(f"Échec de l'encodage pour {image_path} → veuillez réessayer.")
            return False
    except Exception as e:
        print(f"Erreur lors du chargement du visage {image_path} : {e}")
        return False

def save_face(model, known_encodings, known_names, last_face_bbox, last_frame):
    print(f"save_face called with bbox={last_face_bbox} and last_frame={'set' if last_frame is not None else 'None'}")

    if last_face_bbox is None or last_frame is None:
        print("Aucun visage à sauvegarder.")
        return

    try:
        y1, x2, y2, x1 = last_face_bbox
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        margin = 40
        h, w = last_frame.shape[:2]
        top = max(0, y_min - margin)
        bottom = min(h, y_max + margin)
        left = max(0, x_min - margin)
        right = min(w, x_max + margin)
        face_img = last_frame[top:bottom, left:right]
    except Exception as e:
        print(f"Erreur lors du découpage du visage: {e}")
        return

    try:
        cv2.imshow("Crop visage sauvegardé", cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(800)
        cv2.destroyWindow("Crop visage sauvegardé")
    except Exception as e:
        print(f"Erreur lors de l'affichage du visage : {e}")

    name = ask_name()
    if not name:
        print("Aucun nom entré.")
        return

    try:
        os.makedirs("faces", exist_ok=True)
    except Exception as e:
        print(f"Erreur lors de la création du dossier faces : {e}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.jpg"
    path = os.path.join("faces", filename)

    try:
        cv2.imwrite(path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        print(f"Image enregistrée dans {path}")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du fichier : {e}")
        return

    # Recharge proprement le visage enregistré sans recharger toute la base
    success = add_face_to_known(model, known_encodings, known_names, path, name)
    if not success:
        print("Le visage n'a pas été ajouté à la base. Vous pouvez réessayer.")

def is_master_in_frame(model, known_encodings, known_names, last_frame):
    if last_frame is None:
        return False
    frame = last_frame.copy()
    faces = model.get(frame)
    for face in faces:
        emb = face.embedding
        if known_encodings:
            scores = [cosine_similarity(emb, ref) for ref in known_encodings]
            max_score = max(scores)
            best_match = np.argmax(scores)
            if max_score > SIMILARITY_THRESHOLD:
                label = known_names[best_match]
                # Vérifie si le label correspond à un des maîtres
                for master in MODE_MASTER_USERS:
                    if master.lower() in label.lower():
                        return True
    return False
