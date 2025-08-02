import threading
import cv2
from datetime import datetime

from email_utils import send_email
from face_utils import load_known_faces, save_face, is_master_in_frame
from detection import detection_thread_func
from insightface.app import FaceAnalysis
#from voice_control import vocal_mode_control 


# Variables globales partagées
mode_container = ["standard"]  # Liste mutable partagée pour le mode
email_sent = [False]
email_sent_gesture = [False]
gesture_detected_since = [None]

known_encodings = []
known_names = []
last_frame = [None]
annotated_frame = [None]
last_face_bbox = [None]

frame_lock = threading.Lock()

last_seen_times = {}
recent_unknown_encodings = []
sent_unknown_faces = {}

ASSISTANT_LIST = ["SAADIA", "KAIS"]
BLACKLIST = ["VOLEUR", "INTRUS", "BANNI"]
ALLOWED_USERS = ["LASS", "KAÏS", "HAOUA", "SAADIA"]

def set_mode(new_mode):
    mode_container[0] = new_mode
    email_sent[0] = False
    email_sent_gesture[0] = False
    gesture_detected_since[0] = None
    print(f"\n💡 Mode activé : {new_mode.upper()}")

def main():
    global known_encodings, known_names

    model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    model.prepare(ctx_id=-1)

    known_encodings, known_names = load_known_faces(model)

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 20)
    if not cam.isOpened():
        print("Impossible d'ouvrir la webcam.")
        return

    print("\nCaméra ouverte. Appuyez sur :")
    print(" - 'e' pour AJOUTER un visage")
    print(" - 'r' pour recharger la base")
    print(" - 'n' pour mode STANDART")
    print(" - 'v' pour mode VISITEUR")
    print(" - 's' pour mode SURVEILLANCE")
    print(" - 'a' pour mode ASSISTANT")
    print(" - Échap pour quitter\n")

    threading.Thread(
        target=detection_thread_func,
        args=(
            model,
            known_encodings,
            known_names,
            frame_lock,
            annotated_frame,
            last_frame,
            last_face_bbox,
            last_seen_times,
            recent_unknown_encodings,
            sent_unknown_faces,
            ASSISTANT_LIST,
            BLACKLIST,
            ALLOWED_USERS,
            mode_container,
            email_sent,
            email_sent_gesture,
        ),
        daemon=True
    ).start()
    
    # Lancement thread reconnaissance vocale
    #threading.Thread(target=vocal_mode_control, args=(set_mode,), daemon=True).start()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Problème de lecture de la caméra.")
            break

        with frame_lock:
            last_frame[0] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
            display = annotated_frame[0].copy() if annotated_frame[0] is not None else frame.copy()

        cv2.imshow("ai-sha", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('e'):
            if mode_container[0] in ["standard", "surveillance", "visiteur", "assistant"]:
                save_face(model, known_encodings, known_names, last_face_bbox[0], last_frame[0])
            else:
                print("Enregistrement désactivé dans ce mode.")
        elif key == ord('r'):
            known_encodings, known_names = load_known_faces(model)
        elif key == ord('n'):
            if is_master_in_frame(model, known_encodings, known_names, last_frame[0]):
                set_mode("standard")
            else:
                print("⚠️ Aucun visage autorisé détecté pour changer de mode.")
        elif key == ord('v'):
            if is_master_in_frame(model, known_encodings, known_names, last_frame[0]):
                set_mode("visiteur")
            else:
                print("⚠️ Aucun visage autorisé détecté pour changer de mode.")
        elif key == ord('s'):
            if is_master_in_frame(model, known_encodings, known_names, last_frame[0]):
                set_mode("surveillance")
            else:
                print("⚠️ Aucun visage autorisé détecté pour changer de mode.")
        elif key == ord('a'):
            if is_master_in_frame(model, known_encodings, known_names, last_frame[0]):
                set_mode("assistant")
            else:
                print("⚠️ Aucun visage autorisé détecté pour changer de mode.")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
