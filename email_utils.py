import smtplib
from email.message import EmailMessage
import os

def send_email(subject, body, sender, receiver, password, attachment_path=None, mode=None):
    subject_final = f"AI-SHA / MODE : {mode.upper() if mode else 'UNKNOWN'} | {subject}"
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject_final
    msg["From"] = sender
    msg["To"] = receiver

    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            file_data = f.read()
            file_name = os.path.basename(attachment_path)
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            maintype, subtype = "image", "jpeg"
        elif file_name.lower().endswith(".mp4"):
            maintype, subtype = "video", "mp4"
        else:
            maintype, subtype = "application", "octet-stream"
        msg.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=file_name)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        print("E-mail envoyé avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'e-mail : {e}")
