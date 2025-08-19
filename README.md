# Analiză și Clasificare Servicii Site Web

Acest proiect conține două scripturi Python pentru analiza și clasificarea paginilor unui site web:

- **analizare_site.py**: Extrage linkurile din meniul unui site, clasifică paginile (produse, servicii, contact, necunoscut) folosind un model NLP și salvează rezultatele.
- **clasificare_servicii.py**: Analizează paginile de tip "servicii" din rezultatele anterioare, clasifică serviciile în categorii/subcategorii folosind un model BERT și salvează rezultatele detaliate.

---

## Instalare

Activează mediul virtual (Windows):

```
env_activate.bat
```

Instalează dependențele necesare:

```
pip install requests beautifulsoup4 transformers torch
```

---

## Utilizare

### 1. Analiza site-ului

Rulează scriptul pentru a analiza un site și a salva rezultatele:

```
python analizare_site.py
```

Introduceți URL-ul site-ului când vi se solicită. Rezultatele vor fi salvate în `rezultate.json`.

### 2. Clasificarea serviciilor

Asigură-te că ai modelul BERT salvat local (vezi variabila `MODEL_PATH` din script).

Rulează scriptul pentru a clasifica paginile de servicii:

```
python clasificare_servicii.py
```

Rezultatele vor fi salvate în `servicii_categorii.json`.

---

## Structura fișierelor

- `analizare_site.py` – Analizează și clasifică paginile site-ului.
- `clasificare_servicii.py` – Clasifică paginile de servicii în categorii/subcategorii.
- `env_activate.bat` – Activează mediul virtual Python.
- `rezultate.json` – Rezultatele analizei inițiale.
- `servicii_categorii.json` – Rezultatele finale ale clasificării serviciilor.

---

## Cerințe suplimentare

- Pentru `clasificare_servicii.py` este necesar un model BERT salvat local, inclusiv fișierul `id2label.json`.
- Scripturile funcționează pe Windows și folosesc Python 3.8+.

---

## Licență

Acest proiect este destinat uzului