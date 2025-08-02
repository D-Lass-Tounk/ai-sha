import threading
import time
import os
from datetime import datetime, time as dt_time, timedelta
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

from utils import cosine_similarity
from email_utils import send_email

# Constantes pour les mails (facteurs constants)
EMAIL_SENDER = "lasstounk2017@gmail.com"
EMAIL_RECEIVER = "lasstounk2023@gmail.com"
EMAIL_PASSWORD = "pczxinliuvioazxj"

GESTURE_HOLD_SECONDS = 2
SIMILARITY_THRESHOLD = 0.38
MAX_RECENT_UNKNOWN = 50

frame_lock = threading.Lock()
annotated_frame = None
last_frame = None
last_face_bbox = None

mode = None
email_sent = False
email_sent_gesture = False
gesture_detected_since = None
last_seen_times = {}
recent_unknown_encodings = []
sent_unknown_faces = {}
video_frame_buffer = deque(maxlen=2 * 20)  # Ajuster selon VIDEO_PRE_SECONDS * VIDEO_FPS
recording_video = False
video_writer = None
video_save_path = None
last_seen_unknown_face = None

ASSISTANT_LIST = []
BLACKLIST = []
ALLOWED_USERS = []

VIDEO_FPS = 20
MAX_MISSING_SECONDS = 3

# --- CONFIGURATION MODE ASSISTANT AUTOMATIQUE ---
ASSISTANT_START_TIME = dt_time(21,40)   # Début période
ASSISTANT_END_TIME = dt_time(22, 40)     # Fin période
ASSISTANT_CHECK_INTERVAL_MINUTES = 3     # Intervalle entre alertes si absence prolongée
ASSISTANT_WAITING_MINUTES = 2             # Délai d'attente avant envoi de la 1ère alerte

last_assistant_email_time = None          # Dernière alerte envoyée
children_seen_status = {}                  # Statut présence enfants pendant période
assistant_mode_start_time = None          # Heure activation mode assistant
assistant_email_sent_after_delay = False  # Flag envoi alerte après délai

assistant_mode_active = False              # Mode assistant actif ou pas
all_children_home = False                  # Variable indiquant si tous enfants sont à la maison

last_exit_assistant_time = None            # Heure de sortie du mode assistant

def get_next_assistant_start(now=None):
    """
    Calcule la prochaine datetime correspondant au prochain ASSISTANT_START_TIME
    """
    if now is None:
        now = datetime.now()
    today_start = now.replace(hour=ASSISTANT_START_TIME.hour, minute=ASSISTANT_START_TIME.minute, second=0, microsecond=0)
    if now < today_start:
        return today_start
    else:
        # Le start sera demain à la même heure
        return today_start + timedelta(days=1)

def detection_thread_func(
    model,
    known_encodings,
    known_names,
    frame_lock,
    annotated_frame_container,
    last_frame_container,
    last_face_bbox_container,
    last_seen_times,
    recent_unknown_encodings,
    sent_unknown_faces,
    ASSISTANT_LIST,
    BLACKLIST,
    ALLOWED_USERS,
    mode_container,
    email_sent_container,
    email_sent_gesture_container,
):
    global recording_video, video_writer, video_save_path, last_seen_unknown_face, gesture_detected_since
    global last_assistant_email_time, children_seen_status
    global assistant_mode_start_time, assistant_email_sent_after_delay
    global assistant_mode_active, all_children_home, last_exit_assistant_time

    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)

    while True:
        with frame_lock:
            frame_rgb = last_frame_container[0]
            if frame_rgb is None:
                continue
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        now = datetime.now()
        current_time = now.time()

        # --- Reset de la variable all_children_home avant la prochaine plage horaire ---
        if all_children_home:
            if last_exit_assistant_time is not None:
                next_start = get_next_assistant_start(last_exit_assistant_time)
                if now >= next_start - timedelta(minutes=2):  # reset 2 minutes avant le début
                    print("[Auto] Reset all_children_home à False avant nouvelle plage horaire.")
                    all_children_home = False
                    last_exit_assistant_time = None

        # --- Activation/désactivation automatique mode assistant selon plage horaire ET all_children_home ---
        if ASSISTANT_START_TIME <= current_time <= ASSISTANT_END_TIME:
            if not all_children_home and (mode_container[0] != "assistant" or not assistant_mode_active):
                print("[Auto] Activation automatique du mode assistant selon plage horaire.")
                mode_container[0] = "assistant"
                assistant_mode_start_time = now
                assistant_email_sent_after_delay = False
                children_seen_status.clear()
                last_assistant_email_time = None
                assistant_mode_active = True
        else:
            # Fin de plage horaire, désactivation automatique
            if mode_container[0] == "assistant" and assistant_mode_active:
                print("[Auto] Désactivation automatique du mode assistant hors plage horaire.")
                mode_container[0] = "standard"
                print(f"[Auto] Mode actuel : {mode_container[0].upper()}")
                assistant_mode_start_time = None
                children_seen_status.clear()
                last_assistant_email_time = None
                assistant_email_sent_after_delay = False
                assistant_mode_active = False
                all_children_home = False
                last_exit_assistant_time = now

                continue  # saute le reste de la boucle ce tour-ci

        video_frame_buffer.append(frame_bgr.copy())

        result_frame = frame_rgb.copy()
        faces = model.get(frame_rgb)

        anyone_seen = []

        unknown_face_found = False
        unknown_face_embedding = None
        unknown_face_bbox = None

        if faces:
            face_for_bbox = faces[0]
            x1, y1, x2, y2 = face_for_bbox.bbox.astype(int)
            last_face_bbox_container[0] = (y1, x2, y2, x1)
        else:
            last_face_bbox_container[0] = None

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            emb = face.embedding
            label = "Visage inconnu"
            similarity = 0

            if known_encodings:
                scores = [cosine_similarity(emb, ref) for ref in known_encodings]
                max_score = max(scores)
                best_match = np.argmax(scores)
                if max_score > SIMILARITY_THRESHOLD:
                    label = known_names[best_match]
                    similarity = max_score

            if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None:
                for (x, y, _) in face.landmark_3d_68.astype(int):
                    cv2.circle(result_frame, (x, y), 1, (200, 255, 255), -1)

            text = f"{label} ({int(similarity * 100)}%)" if similarity else label
            cv2.putText(result_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (64, 64, 64), 1)

            if label != "Visage inconnu":
                anyone_seen.append(label)

            if any(bname.lower() in label.lower() for bname in BLACKLIST):
                if mode_container[0].lower() in ["standard", "surveillance"] and not email_sent_container[0]:
                    print(f"[ALERTE] Blacklist détectée : {label}")
                    send_email(
                        subject="ALERTE BLACKLIST",
                        body=f"ALERTE : {label} de la blacklist détecté à {now.strftime('%H:%M:%S')}",
                        sender=EMAIL_SENDER,
                        receiver=EMAIL_RECEIVER,
                        password=EMAIL_PASSWORD,
                        mode=mode_container[0]
                    )
                    email_sent_container[0] = True

            if label == "Visage inconnu":
                unknown_face_found = True
                unknown_face_embedding = emb
                unknown_face_bbox = (y1, x2, y2, x1)

                emb_hash = hash(tuple(np.round(emb, decimals=4)))
                last_sent = sent_unknown_faces.get(emb_hash)
                resend_allowed = last_sent is None or (now - last_sent).total_seconds() > 600

                is_new_unknown = all(cosine_similarity(emb, unk) < SIMILARITY_THRESHOLD for unk in recent_unknown_encodings)

                if is_new_unknown:
                    recent_unknown_encodings.append(emb)
                    if len(recent_unknown_encodings) > MAX_RECENT_UNKNOWN:
                        recent_unknown_encodings.pop(0)

        for monitored in ASSISTANT_LIST:
            if monitored.lower() in label.lower():
                last_seen_times[monitored] = now

        # Gestion des gestes "2 doigts"
        results = mp_hands.process(frame_rgb)
        fingers_up = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = [(int(lm.x * frame_rgb.shape[1]), int(lm.y * frame_rgb.shape[0])) for lm in hand_landmarks.landmark]
                finger_ids = [4, 8, 12, 16, 20]
                fingers_up = sum([1 for fid in finger_ids if points[fid][1] < points[fid - 2][1]])

        if fingers_up == 2:
            if gesture_detected_since is None:
                gesture_detected_since = time.time()
            elif time.time() - gesture_detected_since >= GESTURE_HOLD_SECONDS and not email_sent_gesture_container[0]:
                whitelist_present = [label for label in anyone_seen if any(name.lower() in label.lower() for name in ALLOWED_USERS)]
                if whitelist_present:
                    print("[ALERTE] Geste danger détecté.")
                    send_email(
                        subject="ALERTE GESTE DANGER",
                        body=f"Geste danger détecté par : {','.join(whitelist_present)}",
                        sender=EMAIL_SENDER,
                        receiver=EMAIL_RECEIVER,
                        password=EMAIL_PASSWORD,
                        mode=mode_container[0]
                    )
                    email_sent_gesture_container[0] = True
        else:
            gesture_detected_since = None

        # --- Mode Surveillance : gestion vidéo pour visage inconnu ---
        if mode_container[0].lower() == "surveillance":
            print(f"[Surveillance] Mode activé. Visage inconnu détecté : {unknown_face_found}")
            if unknown_face_found:
                last_seen_unknown_face = now
                if not recording_video:
                    print("[Surveillance] Démarrage de l'enregistrement vidéo.")
                    os.makedirs("unknown", exist_ok=True)
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    video_save_path = os.path.join("unknown", f"unknown_{timestamp}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    height, width = frame_bgr.shape[:2]
                    try:
                        video_writer = cv2.VideoWriter(video_save_path, fourcc, VIDEO_FPS, (width, height))
                    except Exception as e:
                        print(f"[Erreur] Création VideoWriter : {e}")
                        video_writer = None
                    else:
                        recording_video = True
                        for f in video_frame_buffer:
                            video_writer.write(f)
                        video_frame_buffer.clear()
                else:
                    print("[Surveillance] Enregistrement vidéo en cours : écriture d'une frame.")
                if video_writer is not None:
                    video_writer.write(frame_bgr)

            if recording_video and last_seen_unknown_face is not None:
                elapsed_missing = (now - last_seen_unknown_face).total_seconds()
                print(f"[Surveillance] Enregistrement actif. Temps depuis dernier visage inconnu : {elapsed_missing:.2f}s")
                if elapsed_missing > MAX_MISSING_SECONDS:
                    print("[Surveillance] Arrêt de l'enregistrement vidéo, envoi de l'alerte email.")
                    if video_writer is not None:
                        video_writer.release()
                    video_writer = None
                    recording_video = False
                    send_email(
                        subject="ALERTE - Visage inconnu détecté",
                        body=f"Un visage inconnu a été détecté à {now.strftime('%H:%M:%S')}. Vidéo jointe.",
                        sender=EMAIL_SENDER,
                        receiver=EMAIL_RECEIVER,
                        password=EMAIL_PASSWORD,
                        attachment_path=video_save_path,
                        mode=mode_container[0]
                    )
                    last_seen_unknown_face = None
                    video_save_path = None

        # --- Mode Assistant : alerte absence automatique avec délai d'attente et intervalle ---
        if mode_container[0].lower() == "assistant":

            # Initialiser children_seen_status si vide
            if not children_seen_status:
                for child in ASSISTANT_LIST:
                    children_seen_status[child] = False

            # Mettre à jour l'état "vu" des enfants détectés (envoi mail **une seule fois**)
            for seen_name in anyone_seen:
                for child in ASSISTANT_LIST:
                    if child.lower() in seen_name.lower() and not children_seen_status.get(child, False):
                        children_seen_status[child] = True
                        print(f"[Assistant] Enfant détecté : {child}. Envoi mail confirmation présence.")
                        send_email(
                            subject=f"Notification Enfant vu : {child}",
                            body=f"L'enfant {child} a été vu pour la première fois dans la période définie.",
                            sender=EMAIL_SENDER,
                            receiver=EMAIL_RECEIVER,
                            password=EMAIL_PASSWORD,
                            mode=mode_container[0]
                        )

            # Vérifier si tous les enfants ont été vus, si oui, sortir du mode assistant
            if all(children_seen_status.values()) and children_seen_status:
                print("[Assistant] Tous les enfants ont été vus. Envoi mail confirmation générale et sortie du mode assistant.")
                send_email(
                    subject="Notification Enfants présents",
                    body="Tous les enfants ont été vus dans la période définie.",
                    sender=EMAIL_SENDER,
                    receiver=EMAIL_RECEIVER,
                    password=EMAIL_PASSWORD,
                    mode=mode_container[0]
                )
                mode_container[0] = "standard"
                print(f"[Assistant] Mode actuel : {mode_container[0].upper()}")
                assistant_mode_start_time = None
                children_seen_status.clear()
                last_assistant_email_time = None
                assistant_email_sent_after_delay = False
                assistant_mode_active = False
                all_children_home = True
                last_exit_assistant_time = now

            else:
                # Gérer délai d'attente avant envoi de la 1ère alerte
                if assistant_mode_start_time is not None:
                    elapsed_since_start = (now - assistant_mode_start_time).total_seconds() / 60
                else:
                    elapsed_since_start = 0

                if not assistant_email_sent_after_delay:
                    if elapsed_since_start >= ASSISTANT_WAITING_MINUTES:
                        absent_children = [child for child, seen in children_seen_status.items() if not seen]
                        if absent_children:
                            body = "Enfants absents :\n" + "\n".join(absent_children)
                            print(f"[Assistant] Envoi 1ère alerte absence après délai d'attente:\n{body}")
                            send_email(
                                subject="Notification Enfants absents",
                                body=body,
                                sender=EMAIL_SENDER,
                                receiver=EMAIL_RECEIVER,
                                password=EMAIL_PASSWORD,
                                mode=mode_container[0]
                            )
                            last_assistant_email_time = now
                            assistant_email_sent_after_delay = True
                        else:
                            # Tous vus, mail confirmation générale + sortie mode assistant
                            print("[Assistant] Tous les enfants ont été vus dès le délai d'attente. Mail confirmation générale.")
                            send_email(
                                subject="Notification Enfants présents",
                                body="Tous les enfants ont été vus dans la période définie.",
                                sender=EMAIL_SENDER,
                                receiver=EMAIL_RECEIVER,
                                password=EMAIL_PASSWORD,
                                mode=mode_container[0]
                            )
                            mode_container[0] = "standard"
                            print(f"[Assistant] Mode actuel : {mode_container[0].upper()}")
                            assistant_mode_start_time = None
                            children_seen_status.clear()
                            last_assistant_email_time = None
                            assistant_email_sent_after_delay = False
                            assistant_mode_active = False
                            all_children_home = True
                            last_exit_assistant_time = now

                else:
                    # Après 1ère alerte, envoyer alertes périodiques toutes les X minutes si enfants toujours absents
                    absent_children = [child for child, seen in children_seen_status.items() if not seen]
                    if absent_children:
                        if last_assistant_email_time is None or (now - last_assistant_email_time).total_seconds() > ASSISTANT_CHECK_INTERVAL_MINUTES * 60:
                            body = "Enfants toujours absents :\n" + "\n".join(absent_children)
                            print(f"[Assistant] Envoi alerte périodique absence:\n{body}")
                            send_email(
                                subject="Notification Enfants absents (rappel)",
                                body=body,
                                sender=EMAIL_SENDER,
                                receiver=EMAIL_RECEIVER,
                                password=EMAIL_PASSWORD,
                                mode=mode_container[0]
                            )
                            last_assistant_email_time = now
                    else:
                        # Tous vus après alertes, mail confirmation générale + sortie mode assistant
                        print("[Assistant] Tous les enfants ont été vus après alertes. Mail confirmation générale et retour mode standard.")
                        send_email(
                            subject="Notification Enfants présents",
                            body="Tous les enfants ont été vus dans la période définie.",
                            sender=EMAIL_SENDER,
                            receiver=EMAIL_RECEIVER,
                            password=EMAIL_PASSWORD,
                            mode=mode_container[0]
                        )
                        mode_container[0] = "standard"
                        print(f"[Assistant] Mode actuel : {mode_container[0].upper()}")
                        assistant_mode_start_time = None
                        children_seen_status.clear()
                        last_assistant_email_time = None
                        assistant_email_sent_after_delay = False
                        assistant_mode_active = False
                        all_children_home = True
                        last_exit_assistant_time = now

        with frame_lock:
            annotated_frame_container[0] = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
