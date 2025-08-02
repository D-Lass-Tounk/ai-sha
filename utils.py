from numpy import dot
from numpy.linalg import norm
import numpy as np

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def embedding_hash(embedding):
    return hash(tuple(np.round(embedding, decimals=4)))


mode_container = None
email_sent = False
email_sent_gesture = False
gesture_detected_since = None

def set_mode(new_mode):
    global mode_container, email_sent, email_sent_gesture, gesture_detected_since
    mode_container[0] = new_mode
    email_sent = False
    email_sent_gesture = False
    gesture_detected_since = None
    print(f"\n💡 Mode activé : {new_mode.upper()}")

    
    
