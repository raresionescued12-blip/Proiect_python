import json, re, warnings, requests, unicodedata, torch, joblib
from bs4 import BeautifulSoup, SoupStrainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

MODEL_PATH = r"C:/Users/RARES LENOVO/Desktop/MUNCA/produse_model/bert-produse-model"
ENCODER_PATH = r"C:/Users/RARES LENOVO/Desktop/MUNCA/produse_model/label_encoder_categorie.pkl"

THRESHOLD_BERT_CONFIDENCE = 0.85
THRESHOLD_MARGIN = 0.20

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

ALLOWED = [
    "carti","televizoare","electronice","haine","cosmetice","jucarii",
    "mobilier","inteligenta artificiala","ingrijire personala","electrocasnice","auto",
    "sport","bricolaj","pet shop","papetarie","sanatate","agricultura","bauturi",
    "alimente","muzica","filme","gaming","bijuterii","accesorii","articole de bucatarie",
    "decoratiuni","articole de camping","instrumente muzicale","papusi","constructii",
    "pictura","modelism","produse traditionale","cadouri","altele"
]

HDRS = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language":"ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
le_cat = joblib.load(ENCODER_PATH)

def _norm(s):
    return unicodedata.normalize("NFD", (s or "")).encode("ascii","ignore").decode().lower().strip()

ALLOWED_MAP = { _norm(c): c for c in ALLOWED }
MODEL_CLASSES_NORM = { _norm(c) for c in le_cat.classes_ }

def extract_text(url):
    try:
        r = requests.get(url, headers=HDRS, timeout=(10,25))
        r.raise_for_status()
        only = SoupStrainer(['title','meta','h1','h2','p','article','section','main'])
        soup = BeautifulSoup(r.content, 'lxml', parse_only=only)
        bits_all, bits_focus = [], []
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        if title:
            bits_all.append(title); bits_focus.append(title)
        for md in soup.find_all("meta", attrs={"name":["description","keywords"]}):
            c = md.get("content")
            if c:
                bits_all.append(c); bits_focus.append(c)
        for h in soup.find_all(["h1","h2"]):
            t = h.get_text(" ", strip=True)
            if t:
                bits_all.append(t); bits_focus.append(t)
        for p in soup.find_all("p"):
            t = p.get_text(" ", strip=True)
            if len(t.split()) >= 4:
                bits_all.append(t)
        text_full = re.sub(r"\s+", " ", " ".join(bits_all)).strip()[:4000]
        text_focus = re.sub(r"\s+", " ", " ".join(bits_focus)).strip()[:800]
        return title, text_full, text_focus
    except Exception as e:
        print(f"[Eroare] {url} -> {e}")
        return "", "", ""

def classify_with_bert(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze(0).cpu()
    top2 = torch.topk(probs, k=2)
    top1_id, top1_p = int(top2.indices[0]), float(top2.values[0])
    top2_id, top2_p = int(top2.indices[1]), float(top2.values[1])
    pred1 = le_cat.inverse_transform([top1_id])[0]
    pred2 = le_cat.inverse_transform([top2_id])[0]
    margin = top1_p - top2_p
    return pred1, pred2, top1_p, margin

def _parse_llm_json(raw):
    raw = raw.strip()
    try:
        return json.loads(raw)
    except:
        try:
            m = re.search(r'\{.*\}', raw, flags=re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except:
            return None
    return None

def _ollama(prompt):
    r = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}},
        timeout=60
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

NON_PRODUCT_PAT = re.compile(
    r"\b(termeni|conditii|condi[tţ]ii|confidentialitate|retur(uri)?|garant(i|ie)|"
    r"livrare|despre noi|contact|ajutor|faq|intrebari frecvente|cookies?|cookie|"
    r"sitemap|harta site|wishlist|cos de cumparaturi|cos|login|autentificare|cont)\b",
    flags=re.IGNORECASE
)

def looks_like_non_product(text_focus):
    return bool(NON_PRODUCT_PAT.search(text_focus or ""))

def quick_map(text):
    t = _norm(text)
    if re.search(r"\bhartie igienica\b", t):
        return "ingrijire personala"
    if re.search(r"\b(pasta de dinti|periut[ae]|apa de gura|ata dentara)\b", t):
        return "ingrijire personala"
    if re.search(r"\b(odorizant|wc\s*net|anticalcar|calcar|degresant|detergent(i)?( de)? (vase|rufe)|bureti|lavete|prosoape hartie|servetele de masa)\b", t):
        return "articole de bucatarie"
    if re.search(r"\b(otet|aceto balsamic)\b", t):
        return "alimente"
    if re.search(r"\b(apa minerala|apa plata|bere|vin(uri)?|ceai|cafea)\b", t):
        return "bauturi"
    if re.search(r"\b(paste|faina|ulei|conserve|fructe|legume|mezeluri|branzeturi|dulciuri|biscuiti)\b", t):
        return "alimente"
    return None

def post_map_category(cat, text):
    t = _norm(text)
    if cat == "cosmetice" and re.search(r"\b(sampon|fixativ|deodorant|gel de dus|crema de fata|creme de corp)\b", t):
        return "ingrijire personala"
    if cat == "bauturi" and re.search(r"\b(otet|aceto)\b", t):
        return "alimente"
    if cat == "altele" and not looks_like_non_product(text):
        m = quick_map(text)
        if m: return m
    return cat

URL_TOKEN_SPLIT = re.compile(r"[-_/]+")
def tokens_from_url(url: str):
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        toks = URL_TOKEN_SPLIT.split(p.path or "") + URL_TOKEN_SPLIT.split(p.netloc or "")
        toks = [ _norm(t) for t in toks if t and t.isascii() ]
        return toks
    except Exception:
        return []

KEYWORD_HINTS = [
    (re.compile(r"\b(gale(t|a|ata|ati|ti)|galeti|galeata|galeți|găleți|găleată|cos(uri)?|coș(uri)?|lazi?|navete?)\b"), "agricultura"),
    (re.compile(r"\b(gradina|gradinarit|ser(a|e)|rasad(uri)?|irigatii|furaje|adapatori|ferm(a|e)|zootehnie)\b"), "agricultura"),
    (re.compile(r"\b(cosuri de rufe|galeata mop|bureti|lavete|detergent)\b"), "articole de bucatarie"),
]

def candidate_cats(text_focus: str, url: str):
    t = _norm(text_focus)
    toks = " ".join(tokens_from_url(url))
    hits = set()
    for rgx, cat in KEYWORD_HINTS:
        if rgx.search(t) or rgx.search(toks):
            hits.add(cat)
    return list(hits)[:8]

def classify_with_llm(text_focus):
    examples = """
Exemple:
- "televizor LED 4K 55 inch" -> televizoare
- "laptop gaming RTX 4060" -> electronice
- "rochie bumbac M" -> haine
- "set bijuterii argint 925" -> bijuterii
- "carte fantasy 320 pagini" -> carti
- "vioara 4/4" -> instrumente muzicale
- "jucarie LEGO Technic" -> jucarii
- "hrana uscata pisici 10kg" -> pet shop
- "racheta tenis" -> sport
- "sampon, pasta de dinti" -> ingrijire personala
- "ruj mat, rimel" -> cosmetice
- "cafea boabe 1kg, vin rosu" -> bauturi
- "paste, faina, ulei, conserve" -> alimente
- "bureti vase, lavete, detergent" -> articole de bucatarie
- "pat tapitat 160x200" -> mobilier
- "bormasina, vopsea" -> bricolaj
- "adeziv gresie" -> constructii
- "consola, jocuri video" -> gaming
- "ghiozdan, caiete, pix" -> papetarie
- "ciocolata artizanala italiana" -> produse traditionale
- "lumanari parfumate" -> decoratiuni
- "cort 3 persoane" -> articole de camping
- "papusa fashion" -> papusi
"""
    rule_nonprod = looks_like_non_product(text_focus)
    guard = "Permite 'altele' doar daca textul este pagina non-produs (wishlist, retur, garantie, contact, cont, login)."
    prompt = f"""
Esti un clasificator e-commerce generic. Raspunde DOAR JSON VALID pe o singura linie cu cheile:
- "category": una din [{", ".join(ALLOWED)}]
- "keywords": lista 3-8 cuvinte cheie din text
- "dataset_hint": fraza scurta in romana (ce exemple sa adaug in dataset ca BERT sa fie mai bun)

Reguli:
- Alege exact o categorie din lista. {guard}
- Daca textul contine termeni de produs (cantitati, marci, materiale, dimensiuni), NU folosi "altele".
{examples}

Text:
\"\"\"{text_focus}\"\"\"
JSON:
"""
    raw = _ollama(prompt)
    data = _parse_llm_json(raw)
    if not data or "category" not in data:
        raw = _ollama(f'Returneaza DOAR JSON compact cu cheile "category","keywords","dataset_hint". Categoriile: {", ".join(ALLOWED)}. Text: """{text_focus}"""')
        data = _parse_llm_json(raw)
        if not data or "category" not in data:
            return None
    cat_norm = _norm(str(data.get("category","")))
    if cat_norm not in ALLOWED_MAP:
        return None
    cat = ALLOWED_MAP[cat_norm]
    if cat == "altele" and not rule_nonprod:
        m = quick_map(text_focus)
        if m: cat = m
    kws = data.get("keywords") or []
    if isinstance(kws, str): kws = [kws]
    kws = [str(x).strip() for x in kws if str(x).strip()]
    hint = (data.get("dataset_hint") or "").strip()
    return {"category": post_map_category(cat, text_focus), "keywords": kws[:8], "dataset_hint": hint[:200]}

def build_hint(final_cat, kws, bert_pred, bert_conf, llm_cat, missing):
    if missing:
        base = f"Categoria {final_cat} nu exista in model. Adauga 50-100 exemple curate pentru {final_cat}"
    else:
        base = f"Adauga exemple suplimentare pentru {final_cat} (BERT conf {bert_conf:.2f}"
        if llm_cat and llm_cat != final_cat:
            base += f", LLM a sugerat {llm_cat}"
        base += ")"
    if kws: base += ": " + ", ".join(kws[:6])
    return base

def main():
    with open("rezultate_llama_copie.json","r",encoding="utf-8") as f:
        pagini = json.load(f)
    rezultate = []
    for i, pagina in enumerate(pagini, start=1):
        if pagina.get("categorie") != "produse":
            continue
        print(f"\n[{i}] Analizez: {pagina['url']}")
        title, text_full, text_focus = extract_text(pagina["url"])
        print(f"  Text extras: {len(text_full)} caractere")
        p1, p2, conf, margin = classify_with_bert(text_full)
        print(f"    BERT: {p1} vs {p2} (conf={conf:.2f}, margin={margin:.2f})")
        if conf >= THRESHOLD_BERT_CONFIDENCE and margin >= THRESHOLD_MARGIN:
            final_cat = post_map_category(p1, text_focus)
            if final_cat == "martisoare":
                final_cat = "altele"
            entry = {
                "site": pagina["site"], "url": pagina["url"], "categorie":"produse",
                "tip_produs": final_cat, "source":"bert",
                "bert_conf": round(conf,4), "bert_margin": round(margin,4),
                "fallback_used": False, "missing_in_model": False, "hint":""
            }
        else:
            print("    Fallback LLM pe text scurt...")
            llm = classify_with_llm(text_focus)
            if llm:
                llm_cat = llm["category"]
                if llm_cat == "martisoare":
                    llm_cat = "altele"
                missing = _norm(llm_cat) not in MODEL_CLASSES_NORM
                final_cat = llm_cat
                hint = llm.get("dataset_hint") or ""
                if missing or not hint:
                    hint = build_hint(final_cat, llm.get("keywords", []), p1, conf, llm_cat, missing)
                print(f"    LLM: {llm_cat} | missing_in_model={missing} | hint: {hint[:120]}")
                entry = {
                    "site": pagina["site"], "url": pagina["url"], "categorie":"produse",
                    "tip_produs": final_cat, "source":"llm",
                    "bert_conf": round(conf,4), "bert_margin": round(margin,4),
                    "fallback_used": True, "missing_in_model": bool(missing), "hint": hint
                }
            else:
                mapped = quick_map(text_focus)
                if mapped:
                    if mapped == "martisoare":
                        mapped = "altele"
                    final_cat = mapped
                    hint = build_hint(final_cat, [], p1, conf, None, _norm(final_cat) not in MODEL_CLASSES_NORM)
                    print(f"    Safety-net lexical: {final_cat} | hint: {hint[:120]}")
                    entry = {
                        "site": pagina["site"], "url": pagina["url"], "categorie":"produse",
                        "tip_produs": final_cat, "source":"lexical",
                        "bert_conf": round(conf,4), "bert_margin": round(margin,4),
                        "fallback_used": True, "missing_in_model": _norm(final_cat) not in MODEL_CLASSES_NORM, "hint": hint
                    }
                else:
                    final_cat = post_map_category(p1, text_focus)
                    if final_cat == "martisoare":
                        final_cat = "altele"
                    hint = build_hint(final_cat, [], p1, conf, None, False)
                    print(f"    LLM invalid -> folosesc BERT: {final_cat} | hint: {hint[:120]}")
                    entry = {
                        "site": pagina["site"], "url": pagina["url"], "categorie":"produse",
                        "tip_produs": final_cat, "source":"bert_fallback",
                        "bert_conf": round(conf,4), "bert_margin": round(margin,4),
                        "fallback_used": True, "missing_in_model": False, "hint": hint
                    }
        print(f"    Categoria finala: {entry['tip_produs']}")
        rezultate.append(entry)
    with open("produse_tipuri.json","w",encoding="utf-8") as f:
        json.dump(rezultate, f, indent=2, ensure_ascii=False)
    print("\nClasificarea salvata in produse_tipuri.json")

if __name__ == "__main__":
    main()
