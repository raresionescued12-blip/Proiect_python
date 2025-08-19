import json
import re
import requests
import subprocess
from bs4 import BeautifulSoup

RE_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
RE_SOCIAL = re.compile(r"https?://(www\.)?(facebook|instagram|linkedin|twitter)\.com/[a-zA-Z0-9_\-./?=&]+", re.I)
RE_PHONE_CAND = re.compile(r'(\+?\d[\d\s().\-]{7,}\d)')

ADDR_KEYWORDS = [
    "strada", "str.", "bulevard", "b-dul", "bd.", "șoseaua", "soseaua", "șos.", "sos.", "calea",
    "aleea", "nr", "numărul", "numarul", "bloc", "bl.", "sc.", "et.", "ap.", "sector",
    "judet", "județ", "localitate", "oraș", "oras", "cod postal", "zip", "romania",
    "bucurești", "bucuresti", "cluj", "timișoara", "timisoara", "iași", "iasi", "sibiu",
    "constanța", "constanta", "olt", "argeș", "arges", "pitesti", "pitești"
]

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" ,;-·\n\t")

def normalize_phone(raw: str) -> str:
    digits = re.sub(r'\D+', '', raw)
    if digits.startswith('00'):
        digits = digits[2:]
    if digits.startswith('0') and len(digits) == 10:
        return '+40' + digits[1:]
    if digits.startswith('40') and len(digits) == 11:
        return '+' + digits
    return ''

def is_ro_phone(norm: str) -> bool:
    return bool(norm) and (norm.startswith('+402') or norm.startswith('+403') or norm.startswith('+407'))

def extract_phones(soup, text: str):
    candidates = set()
    for a in soup.select('a[href^="tel:"]'):
        candidates.add(a.get('href', '')[4:])
    for m in RE_PHONE_CAND.findall(text):
        candidates.add(m)
    for el in soup.find_all(attrs=True):
        for attr, val in el.attrs.items():
            if isinstance(val, str) and any(k in attr.lower() for k in ['data-phone', 'data-tel', 'phone', 'tel']):
                candidates.add(val)
    norms = []
    seen = set()
    for c in candidates:
        n = normalize_phone(c)
        if is_ro_phone(n):
            key = n[-9:]
            if key not in seen:
                seen.add(key)
                norms.append(n)
    return norms

def score_address_line(line: str) -> int:
    l = line.lower()
    score = 0
    for kw in ADDR_KEYWORDS:
        if kw in l:
            score += 2
    length = len(line.strip())
    if 20 <= length <= 180:
        score += 3
    elif length > 180:
        score -= 2
    if line.lower().count("adresa") > 1:
        score -= 2
    return score

def best_address_from_candidates(cands):
    cleaned = list({normalize_spaces(c) for c in cands if c and normalize_spaces(c)})
    filtered = []
    for c in cleaned:
        if not any((c != o and c in o) for o in cleaned):
            filtered.append(c)
    if not filtered:
        return ""
    best = max(filtered, key=score_address_line)
    return best

def collect_address_candidates(soup: BeautifulSoup, full_text: str):
    cands = []
    for tag in soup.find_all(["address"]):
        cands.append(tag.get_text(" ", strip=True))
    for el in soup.find_all(True, attrs={"class": True}):
        cls = " ".join(el.get("class") or []).lower()
        if any(k in cls for k in ["address", "adresa", "locatie", "location", "addr"]):
            cands.append(el.get_text(" ", strip=True))
    for el in soup.find_all(True, attrs={"id": True}):
        i = (el.get("id") or "").lower()
        if any(k in i for k in ["address", "adresa", "locatie", "location", "addr"]):
            cands.append(el.get_text(" ", strip=True))
    for el in soup.select('[itemprop="address"], [itemtype*="PostalAddress"]'):
        cands.append(el.get_text(" ", strip=True))
    for raw in full_text.split("\n"):
        line = normalize_spaces(raw)
        low = line.lower()
        if not line:
            continue
        if any(k in low for k in ADDR_KEYWORDS):
            if not any(bad in low for bad in ["date de contact", "servicii", "contact", "client", "profesionale"]):
                cands.append(line)
    return cands

def fallback_extract_with_ollama(text):
    prompt = f"""
Extrage din textul următor adresa POȘTALĂ principală (nu mai multe!) în ROMÂNĂ.
Răspunde DOAR cu un obiect JSON cu cheia "adresa", fără niciun alt text.

Text:
{text[:4000]}
    """
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3", "--temperature", "0.2"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout.strip()
        json_str = output[output.find("{"):output.rfind("}")+1]
        parsed = json.loads(json_str)
        adresa = parsed.get("adresa", "")
        return normalize_spaces(adresa)
    except Exception as e:
        print(f"[OLLAMA fallback failed] {e}")
        return ""

def extract_contact_info(url):
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(separator="\n")
        telefoane = extract_phones(soup, text)
        emailuri = list(sorted(set(RE_EMAIL.findall(text))))
        if not telefoane or not emailuri:
            prompt = f"""
Extrage din textul următor adresele de email și numerele de telefon, strict JSON:
{{"telefon": [...], "email": [...]}}

Text:
{text[:4000]}
            """
            try:
                result = subprocess.run(
                    ["ollama", "run", "llama3", "--temperature", "0.2"],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                output = result.stdout.strip()
                json_str = output[output.find("{"):output.rfind("}")+1]
                parsed = json.loads(json_str)
                if not telefoane:
                    llm_phones = []
                    for p in parsed.get("telefon", []):
                        n = normalize_phone(p)
                        if is_ro_phone(n):
                            llm_phones.append(n)
                    keyset = set(t[-9:] for t in telefoane)
                    for n in llm_phones:
                        if n[-9:] not in keyset:
                            keyset.add(n[-9:])
                            telefoane.append(n)
                if not emailuri:
                    emailuri = parsed.get("email", [])
            except Exception as e:
                print(f"[OLLAMA fallback (tel/email) failed] {e}")
        social_links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if RE_SOCIAL.match(href):
                social_links.add(href)
        addr_candidates = collect_address_candidates(soup, text)
        adresa_fizica = best_address_from_candidates(addr_candidates)
        if not adresa_fizica:
            adresa_fizica = fallback_extract_with_ollama(text)
        return {
            "telefon": sorted(set(map(normalize_spaces, telefoane))),
            "email": sorted(set(map(normalize_spaces, emailuri))),
            "social_media": sorted(social_links),
            "adresa_fizica": adresa_fizica or ""
        }
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return {
            "telefon": [],
            "email": [],
            "social_media": [],
            "adresa_fizica": ""
        }

def main():
    with open("rezultate_llama_copie.json", "r", encoding="utf-8") as f:
        pagini = json.load(f)
    agregat_pe_site = {}
    for pagina in pagini:
        if pagina.get("categorie") != "contact":
            continue
        site = pagina.get("site")
        url = pagina.get("url")
        print(f"Extragem contact de pe: {url}")
        info = extract_contact_info(url)
        if site not in agregat_pe_site:
            agregat_pe_site[site] = {
                "site": site,
                "url": url,
                "telefon": set(info["telefon"]),
                "email": set(info["email"]),
                "social_media": set(info["social_media"]),
                "adresa_fizica": info["adresa_fizica"]
            }
        else:
            agreg = agregat_pe_site[site]
            agreg["telefon"].update(info["telefon"])
            agreg["email"].update(info["email"])
            agreg["social_media"].update(info["social_media"])
            if not agreg["adresa_fizica"] and info["adresa_fizica"]:
                agreg["adresa_fizica"] = info["adresa_fizica"]
    rezultate = []
    for site, data in agregat_pe_site.items():
        rezultate.append({
            "site": data["site"],
            "url": data["url"],
            "telefon": sorted(data["telefon"]),
            "email": sorted(data["email"]),
            "social_media": sorted(data["social_media"]),
            "adresa_fizica": data["adresa_fizica"]
        })
    with open("contacte_extrase.json", "w", encoding="utf-8") as f:
        json.dump(rezultate, f, indent=2, ensure_ascii=False)
    print("\nDatele de contact au fost salvate în contacte_extrase.json")

if __name__ == "__main__":
    main()
