import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import json
import time
import warnings
from requests.adapters import HTTPAdapter, Retry
from pathlib import Path

from playwright.sync_api import sync_playwright

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

LABELS = ["produse", "servicii", "contact", "necunoscut"]
SKIP_KEYWORDS = ["cookie", "confidentialitate", "retur", "despre", "autor", "politica", "harta-site", ".pdf"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive"
}

DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "root"
DB_PASS = ""
DB_NAME = "webscrapping"

 
HERE = Path(__file__).resolve().parent
F_REZULTATE_LLAMACOPIE = HERE / "rezultate_llama_copie.json"


def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retries = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _norm_full_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if not re.match(r"^https?://", u, re.I):
        u = "http://" + u
    p = urlparse(u)
    path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    norm = f"{p.scheme}://{p.netloc}{path}"
    if p.query:
        norm += f"?{p.query}"
    return norm


def _norm_path_only(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return "/"
    if not re.match(r"^https?://", u, re.I):
        if not u.startswith("/"):
            u = "/" + u
        path = u
    else:
        p = urlparse(u)
        path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return path


def load_ignore_sets_from_db():
    path_prefixes = set()
    exact_urls = set()
    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME
        )
        cur = conn.cursor()
        cur.execute("SELECT adresa FROM `url`")
        rows = cur.fetchall()
        for (adresa,) in rows:
            if not adresa:
                continue
            full_norm = _norm_full_url(adresa)
            path_norm = _norm_path_only(adresa)
            if path_norm != "/":
                path_prefixes.add(path_norm)
            if full_norm:
                exact_urls.add(full_norm)
        cur.close()
        conn.close()
        print(f"[INFO] Încărcat ignore: {len(path_prefixes)} prefixuri, {len(exact_urls)} URL-uri exacte")
    except Exception as e:
        print(f"[WARN] Nu am putut încărca ignore list din DB: {e}")
    return path_prefixes, exact_urls


def should_ignore_url(u: str, path_prefixes: set, exact_urls: set) -> bool:
    full_norm = _norm_full_url(u)
    if full_norm in exact_urls:
        return True
    path = _norm_path_only(u)
    if path == "/":
        return False
    for pref in path_prefixes:
        if path == pref or path.startswith(pref + "/"):
            return True
    return False


def extract_text_from_tag(soup, tag_name):
    tag = soup.find(tag_name)
    return tag.get_text(separator=" ", strip=True) if tag else ""


def get_header_footer_text(url):
    html = fetch_page_content(url)
    if not html:
        return "", ""
    soup = BeautifulSoup(html, "html.parser")
    return extract_text_from_tag(soup, "header"), extract_text_from_tag(soup, "footer")


def fetch_page_content(url):
    """Întâi încearcă requests, dacă e gol sau scurt, fallback pe Playwright"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = requests_retry_session().get(url, headers=HEADERS, timeout=15, verify=False)
            res.raise_for_status()
            if len(res.text.strip()) > 200:
                return res.text
    except Exception:
        pass   

 
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=30000)
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        print(f"[!] Nu am putut încărca cu Playwright {url}: {e}")
        return ""


def clean_page_content(html, header_text, footer_text):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    if header_text:
        text = text.replace(header_text, "")
    if footer_text:
        text = text.replace(footer_text, "")
    return text


def classify_page_with_llama(text, url):
    if "/contact" in url.lower():
        return "contact"
    prompt = f"""
Primești conținutul unei pagini web. Trebuie să identifici categoria paginii, alegând strict una din următoarele:
- produse
- servicii
- contact
- necunoscut

Textul paginii:

\"\"\" 
{text[:3000]} 
\"\"\"

Care este categoria paginii? Răspunde DOAR cu: produse, servicii, contact sau necunoscut.
"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30
        )
        result = response.json()
        raw = result.get("response", "").strip().lower()
        label = next((l for l in LABELS if l in raw), "necunoscut")
        return label
    except Exception as e:
        print(f"[!] Eroare Ollama: {e}")
        return "necunoscut"


def extract_menu_links(root_url, path_prefixes, exact_urls):
    html = fetch_page_content(root_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    menu_tags = soup.select("nav a[href], header a[href], footer a[href]")
    links = set()
    for tag in menu_tags:
        href = tag.get("href")
        if not href or href.startswith("#") or "javascript" in href.lower():
            continue
        full_url = urljoin(root_url, href)
        if urlparse(full_url).netloc != urlparse(root_url).netloc:
            continue
        if any(kw in full_url.lower() for kw in SKIP_KEYWORDS):
            continue
        if should_ignore_url(full_url, path_prefixes, exact_urls):
            print(f" Ignor (din DB): {full_url}")
            continue
        links.add(full_url.split("#")[0])
    return list(links)


def detect_contact_info(text):
    text = text.lower()
    found = 0
    phone_patterns = [
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{3,4}\b",
        r"\b0\d{2}[-.\s]?\d{3}[-.\s]?\d{3,4}\b",
        r"\b\+?40[-.\s]?\d{2}[-.\s]?\d{3}[-.\s]?\d{3,4}\b"
    ]
    for p in phone_patterns:
        if re.search(p, text):
            found += 1
            break
    if re.search(r"\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b", text):
        found += 1
    location_keywords = ["str", "bucurești", "cluj", "jud", "sector", "timisoara", "iasi", "constanta"]
    if any(k in text for k in location_keywords):
        found += 1
    social_media = ["facebook.com", "instagram.com", "linkedin.com", "twitter.com"]
    if any(s in text for s in social_media):
        found += 1
    return found >= 2


def analyze_site(root_url):
    path_prefixes, exact_urls = load_ignore_sets_from_db()
    print(f"\nAnalizez: {root_url}")
    header_text, footer_text = get_header_footer_text(root_url)

    results = []
    combined_text = (header_text + " " + footer_text).strip()
    if detect_contact_info(combined_text):
        results.append({
            "site": root_url,
            "url": root_url + "#footer",
            "categorie": "contact"
        })
        print("  Am gasit informatii de contact în header/footer.")

    links = extract_menu_links(root_url, path_prefixes, exact_urls)
    if not links:
        print(" Nu am găsit linkuri.")
        return results

    for link in links:
        if should_ignore_url(link, path_prefixes, exact_urls):
            print(f" Ignor (din DB): {link}")
            continue
        print(f"[~] Analizez {link}")
        html = fetch_page_content(link)
        if not html:
            print("    Nu am putut încărca pagina.")
            continue
        text = clean_page_content(html, header_text, footer_text)
        if len(text) < 50:
            print("    Text prea scurt.")
            continue
        label = classify_page_with_llama(text, link)
        print(f"    → Categorie: {label}")
        results.append({
            "site": root_url,
            "url": link,
            "categorie": label
        })
        time.sleep(1)
    return results


def save_results(results, filename: Path = F_REZULTATE_LLAMACOPIE):
    filename = Path(filename)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nRezultatele au fost salvate în {filename.name}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1]:
        root_url = sys.argv[1].strip()
    elif sys.stdin.isatty():
        root_url = input("Introdu URL-ul site-ului: ").strip()
    else:
        print("Adauga URL-ul. Ruleaza: python analizare_site_llama_playwright.py <url>", file=sys.stderr)
        sys.exit(2)

    if not root_url.startswith("http://") and not root_url.startswith("https://"):
        root_url = "https://" + root_url

    results = analyze_site(root_url)
    if results:
        save_results(results)   
