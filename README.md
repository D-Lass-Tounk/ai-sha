# 🤖 AI-SHA – Détection Visage & Geste en Temps Réel

**AI-SHA** est un système de surveillance intelligent qui combine la reconnaissance faciale et la détection des gestes de la main. Il permet de :
- Identifier des visages connus via encodage
- Détecter des visages inconnus et les enregistrer
- Compter les doigts levés et détecter des signaux de danger
- Envoyer un e-mail d’alerte en cas de signal dangereux (ex: deux doigts levés)

## 🔧 Fonctionnalités

✅ Reconnaissance faciale avec InsightFace  
✅ Détection des mains avec MediaPipe  
✅ Sauvegarde automatique des visages inconnus  
✅ Enregistrement manuel de visages via interface  
✅ Bip sonore lors d’une détection inconnue  
✅ Envoi automatique d’email d’alerte (via SMTP Gmail)  
✅ Interface simple avec OpenCV

## 🛠️ Technologies

- Python
- OpenCV
- InsightFace
- MediaPipe
- SMTP (Gmail)
- NumPy
- PIL
- Tkinter

## ✉️ Alerte de danger

Un geste à deux doigts est interprété comme un **signal d'urgence**.  
Lorsqu’il est détecté :
- Un email est envoyé automatiquement
- Une alerte est affichée dans la console

## 📸 Exemple d’utilisation

```bash
- 's' pour enregistrer un visage
- 'r' pour recharger la base
- 'ECHAP' pour quitter
```

## 📁 Organisation du projet

```
ai-sha/
│
├── faces/                # Visages connus
├── unknown/              # Visages inconnus détectés
├── main.py               # Script principal
├── README.md             # Ce fichier
└── AI-SHA_Version_7.pdf  # Document de présentation
```

## 📬 Configuration de l’e-mail

Crée une application Gmail avec mot de passe spécifique :  
[https://myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)  
Et insère les infos dans ton script :
```python
sender = "ton_email@gmail.com"
receiver = "destinataire@gmail.com"
password = "mot_de_passe_application"
```

## 🧑‍💻 Auteur

**Lassana Tounkara**  
📧 lassanatounkara.dev@gmail.com  
🔗 [GitHub: D-Lass-Tounk](https://github.com/D-Lass-Tounk)