# StadtIdee

Eine Streamlit-App für Stadtideen: Ein Foto eines öffentlichen Orts hochladen oder direkt aufnehmen, Ergänzungen einzeichnen, vom Teachable-Machine-Modell erkennen lassen und den Vorschlag veröffentlichen.

## Lokal starten (Ubuntu)

```bash
sudo apt install python3-venv
cd ~/Downloads/StadtIdee-GitHub
bash start_app.sh
```

Das Startskript erstellt bei Bedarf die virtuelle Umgebung `.venv`, installiert die Abhängigkeiten und startet Streamlit. Öffne anschließend die im Terminal genannte Adresse, normalerweise `http://localhost:8501`.

Alternativ kannst du die Umgebung manuell aufsetzen:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Fotos aufnehmen

Unter **Eigene Projekte** kannst du **Foto aufnehmen** auswählen. Erlaube dafür den Kamerazugriff im Browser. Auf `localhost` funktioniert die Kamerafreigabe direkt; für eine veröffentlichte Website benötigt der Browser HTTPS.

## Repository-Struktur

```text
StadtIdee-GitHub/
├── app.py
├── classifier.py
├── image_processing.py
├── project_store.py
├── ui_theme.py
├── requirements.txt
├── runtime.txt
├── start_app.sh
├── .gitignore
├── README.md
└── model/
    ├── keras_model.h5
    └── labels.txt
```

Beide Dateien unter `model/` werden für die Bilderkennung benötigt. Die Projekt-Datenbank `submitted_projects.sqlite3` wird beim Start automatisch erzeugt. Trainingsdaten, virtuelle Umgebungen, Datenbanken und Zugangsdaten sind nicht Teil dieses App-Exports.

## Dateien zu GitHub hochladen

Öffne ein Terminal im Exportordner und ersetze die Platzhalter durch deinen GitHub-Benutzernamen und Repository-Namen:

```bash
cd ~/Downloads/StadtIdee-GitHub
git init
git add .
git commit -m "StadtIdee Streamlit App"
git branch -M main
git remote add origin https://github.com/DEIN-NAME/DEIN-REPOSITORY.git
git push -u origin main
```

Anschließend kannst du das Repository mit Streamlit Community Cloud verbinden und `app.py` als Hauptdatei auswählen. `runtime.txt` gibt Python 3.11 für das Hosting vor.
