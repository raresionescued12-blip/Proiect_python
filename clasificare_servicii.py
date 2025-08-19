import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import re
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "C:/Users/RARES LENOVO/Desktop/MUNCA/model_servicii"
model_cat = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
le_cat = joblib.load(f"{model_path}/label_encoder_categorie.pkl")

CATEGORII_VALID = list(le_cat.classes_)
THRESHOLD_BERT_CONFIDENCE = 0.5

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_OPTIONS = {"temperature": 0.0, "num_predict": 48}

ACRONYMS = {"AI", "SEO", "CRM", "ERP", "PPC", "SaaS", "UI", "UX"}

GENERIC_BAD = {
    "servicii", "service", "general", "diverse",
    "noutăți", "noutati", "blog", "contact", "despre noi", "acasa", "acasă"
}

DIAC_MAP = {"ă":"a","â":"a","î":"i","ș":"s","ş":"s","ț":"t","ţ":"t",
            "Ă":"A","Â":"A","Î":"I","Ș":"S","Ş":"S","Ț":"T","Ţ":"T"}

def strip_diac(s:str)->str:
    return "".join(DIAC_MAP.get(ch, ch) for ch in (s or ""))

def clean_one_line(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\n"," ").replace("\r"," ")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \"'.,;:()[]{}")

def titlecase_ro(s: str) -> str:
    if not s: return s
    parts = re.split(r'(\s+|[-/])', s)
    out = []
    for p in parts:
        if not p or re.fullmatch(r'(\s+|[-/])', p):
            out.append(p); continue
        if p.upper() in ACRONYMS:
            out.append(p.upper())
        else:
            out.append(p[:1].upper() + p[1:].lower())
    return "".join(out).strip()

def is_generic(s: str) -> bool:
    key = strip_diac(s).lower().strip()
    return key in GENERIC_BAD

def validate_subcat(s: str) -> str:
    s = clean_one_line(s)
    if not s:
        return "necunoscut"
    s = re.split(r'[|:;•·<>]', s)[0].strip()
    words = s.split()
    if len(words) > 4:
        s = " ".join(words[:4])
    if len(s) > 50:
        s = s[:50].rsplit(" ", 1)[0]
    s = titlecase_ro(s)
    if is_generic(s) or not s:
        return "necunoscut"
    return s

def predict_category(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model_cat(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        max_prob, pred_id = torch.max(probs, dim=1)
        pred_label = le_cat.inverse_transform([pred_id.item()])[0]
    return pred_label, max_prob.item()

def extract_text(url):
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        parts = []
        if soup.title and soup.title.string:
            parts.append(clean_one_line(soup.title.string))
        h1 = soup.find("h1")
        if h1:
            parts.append(clean_one_line(h1.get_text(" ", strip=True)))
        md = soup.find("meta", attrs={"name":"description"})
        if md and md.get("content"):
            parts.append(clean_one_line(md["content"]))
        ogt = soup.find("meta", property="og:title")
        if ogt and ogt.get("content"):
            parts.append(clean_one_line(ogt["content"]))
        seen, out = set(), []
        for p in parts:
            if p and p not in seen:
                seen.add(p); out.append(p)
        return " | ".join(out)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def ollama_request(prompt: str, expect_json: bool = False, timeout: int = 45) -> str:
    payload = {"model": "llama3", "prompt": prompt, "stream": False, "options": OLLAMA_OPTIONS}
    if expect_json:
        payload["format"] = "json"
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()

def ollama_category(text):
    prompt = (
        f"Ești un clasificator strict.\n"
        f"Text: {text}\n\n"
        f"Alege categoria exactă din această listă: {', '.join(CATEGORII_VALID)}.\n"
        f"Răspunde STRICT în format JSON pe o singură linie: "
        f'{{"category":"<valoare din listă sau necunoscuta>"}}'
    )
    try:
        raw = ollama_request(prompt, expect_json=True)
        cat = "necunoscuta"
        try:
            data = json.loads(raw)
            val = str(data.get("category","")).strip()
            if val in CATEGORII_VALID: cat = val
            elif val.lower() == "necunoscuta": cat = "necunoscuta"
        except Exception:
            line = clean_one_line(raw)
            if line in CATEGORII_VALID: cat = line
            elif line.lower() == "necunoscuta": cat = "necunoscuta"
        return cat
    except Exception as e:
        print(f"Ollama API error (category): {e}")
        return "necunoscuta"

def ollama_subcategory(text, categoria_principala):
    prompt = (
        "Ești atent la detalii și răspunzi concis.\n"
        f"Text: {text}\n"
        f"Categoria principală: {categoria_principala}\n\n"
        "Extrage subcategoria *cea mai specifică* în limba română, cu diacritice, fără numele firmei.\n"
        "Interzis: răspunsuri generice (Servicii, General, Contact, Blog etc.).\n"
        "Lungime: 1–4 cuvinte. Fără ghilimele, fără explicații.\n"
        'Răspunde STRICT în JSON: {"subcategory": "<șir scurt>"}'
    )
    try:
        raw = ollama_request(prompt, expect_json=True)
        sub = ""
        try:
            data = json.loads(raw)
            sub = str(data.get("subcategory",""))
        except Exception:
            sub = raw
        sub = validate_subcat(sub)
        return sub
    except Exception as e:
        print(f"Ollama API error (subcategory): {e}")
        return "necunoscut"

def main():
    with open("rezultate_llama_copie.json", "r", encoding="utf-8") as f:
        pages = json.load(f)

    results = []

    for page in pages:
        if page.get("categorie") == "servicii":
            url = page["url"]
            text = extract_text(url)
            if not text:
                results.append({
                    "url": url,
                    "categorie_originala": page["categorie"],
                    "categorie_prezisa": "text not found",
                    "subcategorie_prezisa": "",
                    "text_extras": ""
                })
                continue

            pred_cat_bert, prob_bert = predict_category(text)

            if pred_cat_bert not in CATEGORII_VALID:
                pred_cat_bert = "necunoscuta"

            if prob_bert < THRESHOLD_BERT_CONFIDENCE or pred_cat_bert == "necunoscuta":
                pred_cat_ollama = ollama_category(text)
                final_cat = pred_cat_ollama if pred_cat_ollama != "necunoscuta" else pred_cat_bert
            else:
                final_cat = pred_cat_bert

            subcat = ""
            if final_cat != "necunoscuta":
                subcat = ollama_subcategory(text, final_cat)

            results.append({
                "url": url,
                "categorie_originala": page["categorie"],
                "categorie_prezisa": final_cat,
                "subcategorie_prezisa": subcat,
                "text_extras": text,
                "bert_confidence": prob_bert
            })

    with open("servicii_subcategorii.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
