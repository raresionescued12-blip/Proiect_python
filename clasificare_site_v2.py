
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import re
import difflib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import joblib
from collections import Counter
import mysql.connector
from mysql.connector import Error

DEFAULT_DEPTH = 2            
DEFAULT_MIN_CONF = 0.7       
AUTO_DEEPEN = True           
AUTO_DEEPEN_DEPTH = 3       
MIN_TEXT_LEN = 400  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "C:/Users/RARES LENOVO/Desktop/MUNCA/model_servicii"
model_cat = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
le_cat = joblib.load(f"{model_path}/label_encoder_categorie.pkl")

CATEGORII_VALID = list(le_cat.classes_)
THRESHOLD_BERT_CONFIDENCE = 0.5

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_OPTIONS = {"temperature": 0.0, "num_predict": 128}

DB = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "webscrapping",
}

ACRONYMS = {"AI","SEO","CRM","ERP","PPC","SaaS","UI","UX"}
GENERIC_BAD = {"servicii","service","general","diverse","noutati","blog","contact","despre noi","acasa"}
DIAC_MAP = {"ă":"a","â":"a","î":"i","ș":"s","ş":"s","ț":"t","ţ":"t","Ă":"A","Â":"A","Î":"I","Ș":"S","Ş":"S","Ț":"T","Ţ":"T"}
RO_STOP = {
    "si","sau","de","din","la","cu","pe","in","pentru","fara","un","o","ale","al","ai","este","sunt","noi","voi","ei","ele","ce","care","despre",
    "firma","companie","oferta","oferim","servicii","produse","home","acasa","contact","despre","politica","cookie","gdpr","termeni","conditii","pagina","site","web"
}

def strip_diac(s:str)->str:
    return "".join(DIAC_MAP.get(ch, ch) for ch in (s or ""))

def normalize(s: str) -> str:
    return strip_diac((s or "")).lower().strip()

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
    key = normalize(s)
    return key in GENERIC_BAD

def validate_subcat(s: str) -> str:
    s = clean_one_line(s)
    if not s: return "necunoscut"
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

def fetch_raw_corrections():
    rows = []
    try:
        cnx = mysql.connector.connect(**DB)
        cur = cnx.cursor()
        cur.execute("SELECT `text` FROM `subcategory_corrections`")
        for (txt,) in cur.fetchall():
            if txt:
                rows.append(str(txt))
        cur.close()
        cnx.close()
    except Error as e:
        print(f"[WARN] DB error while reading subcategory_corrections: {e}")
    return rows

def extract_urls(s: str):
    if not s: return []
    return re.findall(r'https?://\S+', s, flags=re.I)

def parse_correction_text(t: str):
    if not t: 
        return []
    s = clean_one_line(t)
    urls = extract_urls(s)
    s_wo_urls = re.sub(r'https?://\S+', ' ', s, flags=re.I)
    url_host, url_path = None, None
    if urls:
        try:
            u = urlparse(urls[0])
            url_host = (u.hostname or "").lower()
            url_host = re.sub(r'^www\.', '', url_host)
            url_path = u.path or ""
        except Exception:
            pass
    rules = []
    def add_rule(w, c):
        wv, cv = validate_subcat(w), validate_subcat(c)
        if wv and wv.lower() != "necunoscut" and cv:
            rules.append({"wrong": wv, "correct": titlecase_ro(cv), "host": url_host, "path": url_path})
    for m in re.finditer(r'ai\s+scris\s+(.+?)\s+(?:si\s+)?(?:corect\s+este|trebuie|ar\s+trebui\s+sa\s+fie|se\s+corecteaza\s+in|devine|foloseste|forma\s+corecta|standard)\s+(.+)$', s_wo_urls, flags=re.I):
        add_rule(m.group(1), m.group(2))
    for m in re.finditer(r'scris\s+(.+?)\s+si\s+trebuie\s+(.+)$', s_wo_urls, flags=re.I):
        add_rule(m.group(1), m.group(2))
    quoted = re.findall(r'[„“"\'`](.+?)[”"\'`]', s_wo_urls)
    if len(quoted) >= 2:
        add_rule(quoted[0], quoted[1])
    for m in re.finditer(r'["“\'`]?([^"“\'`]+?)["”\'`]?\s*(?:->|=>|=|:)\s*["“\'`]?([^"”\'`]+?)["”\'`]?(?:$|[,;\|])', s_wo_urls, flags=re.I):
        add_rule(m.group(1), m.group(2))
    for m in re.finditer(r'gresit\s*[:\-]\s*([^,;\|]+?)\s*,\s*corect\s*[:\-]\s*([^,;\|]+)', s_wo_urls, flags=re.I):
        add_rule(m.group(1), m.group(2))
    return rules

def build_corrections_map():
    raw_rows = fetch_raw_corrections()
    rules = []
    for line in raw_rows:
        rules.extend(parse_correction_text(line))
    uniq = {}
    for r in rules:
        key = (normalize(r["wrong"]), r.get("host") or "", r.get("path") or "")
        if key not in uniq:
            uniq[key] = r
    rules = list(uniq.values())
    print(f"[INFO] Loaded {len(rules)} subcategory correction rules.")
    return rules

def _norm_free(s: str) -> str:
    s = strip_diac((s or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _approx_contains(text_norm: str, needles, min_ratio: float = 0.86) -> bool:
    for n in needles:
        n = _norm_free(n)
        if not n:
            continue
        if n in text_norm:
            return True
        L = len(n)
        if L < 4:
            continue
        for i in range(0, max(1, len(text_norm) - L + 1)):
            window = text_norm[i:i+L+2]
            if difflib.SequenceMatcher(None, n, window).ratio() >= min_ratio:
                return True
    return False

def _extract_host_loose(s: str):
    urls = extract_urls(s)
    if urls:
        try:
            u = urlparse(urls[0])
            host = (u.hostname or "").lower()
            host = re.sub(r'^www\.', '', host)
            return host, (u.path or "")
        except Exception:
            pass
    m = re.search(r'(?:pentru|pt\.?|site-?ul|siteul|la|despre)\s+([a-z0-9\.\-]+\.[a-z]{2,})', s, flags=re.I)
    if m:
        h = m.group(1).lower()
        h = re.sub(r'^www\.', '', h)
        return h, ""
    m = re.search(r'(?:domeniul|pe)\s+([a-z0-9\.\-]+\.[a-z]{2,})', s, flags=re.I)
    if m:
        h = m.group(1).lower()
        h = re.sub(r'^www\.', '', h)
        return h, ""
    return None, ""

def parse_site_directives_text(t: str):
    if not t:
        return []
    raw = clean_one_line(t)
    s = _norm_free(raw)
    host, path = _extract_host_loose(raw)
    if not host:
        return []
    REQUERY_WORDS = [
        "mai cauta informatii","mai cauta informații","cauta mai mult","cauta mai multe",
        "reanalizeaza","reanalizează","reanlizeaza","reia analiza",
        "nu ai facut bine","nu e bine","nu este bine","nu-i bine",
        "mai sapa","sapa mai adanc","sapă mai adânc","investigheaza","detaliaza",
        "completeaza analiza","mergi mai in detaliu","mai in profunzime","mai profund",
        "ai ratat","lipsesc informatii","lispesc informatii","nu ai gasit destule",
        "revizuieste","revizuiește","corecteaza analiza","imbunatateste","îmbunătățește"
    ]
    STRICTER_THRESHOLD_WORDS = [
        "fii mai strict","mai strict","ridica pragul","creste pragul",
        "accepta doar sigur","doar daca esti sigur","confidence mai mare",
        "incredere mai mare","minconf mai mare"
    ]
    FORCE_OLLAMA_WORDS = [
        "ignora bert","foloseste doar ollama","numai ollama","force ollama","forteaza ollama"
    ]
    DEPTH_UP_WORDS = [
        "intra mai adanc","mai multe pagini","urmaresti linkuri","navigheaza intern",
        "crawleaza putin","mergi 2 clickuri","2 clickuri","treci prin subpagini"
    ]
    flags = {
        "requery": False,
        "depth": 1,
        "min_conf": None,
        "force_ollama": False,
        "notes": None,
        "host": host,
        "path": path or ""
    }
    if _approx_contains(s, REQUERY_WORDS) or _approx_contains(s, DEPTH_UP_WORDS):
        flags["requery"] = True
        flags["depth"] = max(flags["depth"], 2)
    if _approx_contains(s, STRICTER_THRESHOLD_WORDS):
        flags["min_conf"] = 0.7
    if _approx_contains(s, FORCE_OLLAMA_WORDS):
        flags["force_ollama"] = True
    m = re.search(r'\bdepth\s*=\s*(\d+)\b', raw, flags=re.I)
    if m:
        try:
            flags["depth"] = max(1, int(m.group(1)))
        except Exception:
            pass
    m = re.search(r'\bmin[_\- ]?conf\s*=\s*(0(?:\.\d+)?|1(?:\.0+)?)\b', raw, flags=re.I)
    if m:
        try:
            flags["min_conf"] = float(m.group(1))
        except Exception:
            pass
    m = re.search(r'\bnotes?\s*:\s*(.+)$', raw, flags=re.I)
    if m:
        flags["notes"] = m.group(1).strip()
    return [flags]

def build_site_directives_map():
    raw_rows = fetch_raw_corrections()
    directives = []
    for line in raw_rows:
        directives.extend(parse_site_directives_text(line))
    by_host = {}
    for d in directives:
        h = d["host"]
        by_host[h] = d
    if by_host:
        print(f"[INFO] Loaded site-level directives for {len(by_host)} hosts.")
    return by_host

def ollama_request(prompt: str, expect_json: bool = False, timeout: int = 45) -> str:
    payload = {"model":"llama3","prompt":prompt,"stream":False,"options":OLLAMA_OPTIONS}
    if expect_json:
        payload["format"] = "json"
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

def predict_category(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model_cat(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        max_prob, pred_id = torch.max(probs, dim=1)
        pred_label = le_cat.inverse_transform([pred_id.item()])[0]
    return pred_label, max_prob.item()

def extract_ldjson_snippets(soup: BeautifulSoup):
    out = []
    for tag in soup.find_all("script", type=lambda t: t and "ld+json" in t.lower()):
        try:
            txt = tag.string or tag.get_text() or ""
            data = json.loads(txt)
            def flatten(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k.lower() in {"name","description","servicetype","category","offers","brand","areaserved"}:
                            if isinstance(v, (str,int,float)):
                                yield str(v)
                            elif isinstance(v, (list,dict)):
                                yield from flatten(v)
                        else:
                            if isinstance(v, (list,dict)):
                                yield from flatten(v)
                elif isinstance(obj, list):
                    for it in obj:
                        yield from flatten(it)
            snippets = [clean_one_line(s) for s in flatten(data)]
            snippets = [s for s in snippets if s]
            if snippets:
                out.append(" | ".join(dict.fromkeys(snippets)))
        except Exception:
            continue
    return out

def extract_meta_and_heads(soup: BeautifulSoup):
    parts = []
    if soup.title and soup.title.string:
        parts.append(clean_one_line(soup.title.string))
    h1 = soup.find("h1")
    if h1:
        parts.append(clean_one_line(h1.get_text(" ", strip=True)))
    for name in ["description","keywords"]:
        md = soup.find("meta", attrs={"name":name})
        if md and md.get("content"):
            parts.append(clean_one_line(md["content"]))
    for prop in ["og:title","og:description","twitter:title","twitter:description"]:
        og = soup.find("meta", property=prop) or soup.find("meta", attrs={"name":prop})
        if og and og.get("content"):
            parts.append(clean_one_line(og.get("content")))
    return parts

def extract_keywords_from_url(url: str):
    p = urlparse(url)
    host = p.hostname or ""
    path = unquote(p.path or "")
    toks = re.split(r"[^\w\-]+", strip_diac(host + " " + path))
    toks = [t.lower().strip("-_") for t in toks if t and len(t) > 2]
    toks = [t for t in toks if t not in RO_STOP]
    toks = [t for t in toks if not re.fullmatch(r"\d{2,}", t)]
    return list(dict.fromkeys(toks))[:12]

def extract_text(url, depth=1, visited=None):
    if visited is None:
        visited = set()
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        parts = []
        parts += extract_meta_and_heads(soup)
        parts += extract_ldjson_snippets(soup)
        for h in soup.find_all(["h1","h2"]):
            parts.append(clean_one_line(h.get_text(" ", strip=True)))
        p_count = 0
        for p in soup.find_all("p"):
            txt = clean_one_line(p.get_text(" ", strip=True))
            if txt:
                parts.append(txt)
                p_count += 1
                if p_count >= 6:
                    break
        li_count = 0
        for li in soup.find_all("li"):
            txt = clean_one_line(li.get_text(" ", strip=True))
            if txt and 5 <= len(txt) <= 140:
                parts.append(txt)
                li_count += 1
                if li_count >= 10:
                    break
        seen, out = set(), []
        for p in parts:
            if p and p not in seen:
                seen.add(p); out.append(p)
        main_text = " | ".join(out)
        kw = extract_keywords_from_url(url)
        if kw:
            main_text = (" | ".join(kw) + " | " + main_text).strip(" |")
        if depth > 1:
            base_host = urlparse(url).netloc
            base_host = re.sub(r"^www\.", "", base_host.lower())
            candidates = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("#"):
                    continue
                try:
                    full = requests.compat.urljoin(url, href)
                except Exception:
                    continue
                u = urlparse(full)
                if not u.scheme.startswith("http"):
                    continue
                host = re.sub(r"^www\.", "", (u.hostname or "").lower())
                if host != base_host:
                    continue
                path = (u.path or "").lower()
                if any(k in path for k in ["servici","servii","solut","ofer","portofol","competent","expertiz"]):
                    candidates.append(full)
            follow = []
            seen_paths = set()
            for c in candidates:
                up = urlparse(c)
                pth = up.path or ""
                if pth in seen_paths:
                    continue
                seen_paths.add(pth)
                follow.append(c)
                if len(follow) >= min(3, depth * 2):
                    break
            visited.add(url)
            for nxt in follow:
                if nxt in visited:
                    continue
                sub = extract_text(nxt, depth=1, visited=visited)
                if sub:
                    main_text = (main_text + " | " + sub)[:20000]
        return main_text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def ollama_category(text):
    prompt = (
        f"Esti un clasificator strict.\n"
        f"Text: {text}\n\n"
        f"Alege categoria exacta din aceasta lista: {', '.join(CATEGORII_VALID)}.\n"
        f"Raspunde STRICT in format JSON pe o singura linie: "
        f'{{"category":"<valoare din lista sau necunoscuta>"}}'
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
        "Esti atent la detalii si raspunzi concis.\n"
        f"Text: {text}\n"
        f"Categoria principala: {categoria_principala}\n\n"
        "Extrage subcategoria cea mai specifica in limba romana, cu diacritice, fara numele firmei.\n"
        "Interzis: raspunsuri generice (Servicii, General, Contact, Blog etc.).\n"
        "Lungime: 1-4 cuvinte. Fara ghilimele, fara explicatii.\n"
        'Raspunde STRICT in JSON: {"subcategory": "<sir scurt>"}'
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

def summarize_site(all_texts: list, counts: dict, host: str):
    combined = " | ".join(all_texts)[:18000]
    heuristic = None
    if counts:
        heuristic = max(counts.items(), key=lambda kv: kv[1])[0]
    prompt = (
        "Rezum concis, in 3-4 fraze, ce ofera compania pe baza textelor de pe site.\n"
        "Evita bulleturi, include domeniul/nisa si tipurile de servicii predominante.\n"
        f"Texte: {combined}\n\n"
        "Raspunde in limba romana, fara prefixe."
    )
    try:
        summary = ollama_request(prompt, expect_json=False, timeout=60)
        summary = clean_one_line(summary)
    except Exception:
        summary = ""
    return heuristic, summary

def infer_site_from_url_list(pages):
    hosts = [urlparse(p.get("url","")).hostname or "" for p in pages if p.get("url")]
    hosts = [h for h in hosts if h]
    if not hosts: return ""
    bare = hosts[0].lower()
    bare = re.sub(r"^www\.", "", bare)
    return bare

def apply_subcat_correction(predicted_subcat: str, rules: list, page_url: str) -> str:
    if not predicted_subcat:
        return predicted_subcat
    key = normalize(predicted_subcat)
    host, path = None, None
    try:
        u = urlparse(page_url or "")
        host = (u.hostname or "").lower()
        host = re.sub(r"^www\.", "", host)
        path = u.path or ""
    except Exception:
        pass
    for r in rules:
        if normalize(r["wrong"]) != key:
            continue
        if r.get("host"):
            if r["host"] != (host or ""):
                continue
            if r.get("path") and r["path"] and r["path"] not in (path or ""):
                continue
            return validate_subcat(r["correct"])
    for r in rules:
        if not r.get("host") and normalize(r["wrong"]) == key:
            return validate_subcat(r["correct"])
    return predicted_subcat

def main():
    input_json = "rezultate_llama_copie.json"
    output_json = "servicii_subcategorii.json"
    correction_rules = build_corrections_map()
    site_directives = build_site_directives_map()
    with open(input_json, "r", encoding="utf-8") as f:
        pages = json.load(f)
    per_page = []
    all_texts = []
    cat_counts = Counter()
    for page in pages:
        if page.get("categorie") != "servicii":
            continue
        url = page.get("url")
        if not url:
            continue
        host = ""
        try:
            u = urlparse(url)
            host = (u.hostname or "").lower()
            host = re.sub(r"^www\.", "", host)
        except Exception:
            pass
        d = site_directives.get(host, {}) if host else {}
        local_depth = int(d.get("depth", 1) or 1)
        if d.get("requery"):
            local_depth = max(local_depth, 2)
        force_ollama = bool(d.get("force_ollama", False))
        local_min_conf = d.get("min_conf", None)
        if local_min_conf is not None:
            try:
                local_min_conf = float(local_min_conf)
            except Exception:
                local_min_conf = None
        text = extract_text(url, depth=local_depth)
        if not text:
            per_page.append({
                "url": url,
                "categorie_originala": page.get("categorie",""),
                "categorie_prezisa": "text not found",
                "subcategorie_prezisa": "",
                "text_extras": "",
                "bert_confidence": 0.0,
                "directives": d or None
            })
            continue
        if not force_ollama:
            pred_cat_bert, prob_bert = predict_category(text)
            if pred_cat_bert not in CATEGORII_VALID:
                pred_cat_bert = "necunoscuta"
        else:
            pred_cat_bert, prob_bert = ("necunoscuta", 0.0)
        min_conf_to_use = local_min_conf if (local_min_conf is not None) else THRESHOLD_BERT_CONFIDENCE
        if force_ollama or prob_bert < min_conf_to_use or pred_cat_bert == "necunoscuta":
            pred_cat_ollama = ollama_category(text)
            final_cat = pred_cat_ollama if pred_cat_ollama != "necunoscuta" else pred_cat_bert
        else:
            final_cat = pred_cat_bert
        subcat = ""
        if final_cat != "necunoscuta":
            subcat = ollama_subcategory(text, final_cat)
            subcat = apply_subcat_correction(subcat, correction_rules, url)
        per_page.append({
            "url": url,
            "categorie_originala": page.get("categorie",""),
            "categorie_prezisa": final_cat,
            "subcategorie_prezisa": subcat,
            "text_extras": text,
            "bert_confidence": prob_bert,
            "directives": d or None
        })
        if final_cat and final_cat != "necunoscuta":
            cat_counts[final_cat] += 1
        all_texts.append(text)
    site_host = infer_site_from_url_list(per_page)
    dominant, summary = summarize_site(all_texts, cat_counts, site_host)
    out = {
        "pages": per_page,
        "site_summary": {
            "site": site_host,
            "dominant_category": dominant or "necunoscuta",
            "categorii_numar": dict(cat_counts),
            "descriere": summary
        }
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()

