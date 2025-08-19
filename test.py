import re, sys, time
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

def canon_variants(u: str):
    u = u.strip()
    if not re.match(r'^https?://', u, re.I):
        u = 'https://' + u
    p = urlparse(u)
    host = (p.netloc or '').lower()
    bare = host[4:] if host.startswith('www.') else host
    base_https = f"https://{bare}"
    base_http  = f"http://{bare}"
    return [
        base_https, base_https + "/",
        base_http,  base_http  + "/",
        f"https://www.{bare}", f"https://www.{bare}/",
        f"http://www.{bare}",  f"http://www.{bare}/",
    ]

def fetch_first_ok(urls, timeout=20):
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
        "Upgrade-Insecure-Requests": "1",
    })
    for u in urls:
        try:
            r = sess.get(u, timeout=timeout, allow_redirects=True)
            print(f"[HTTP] GET {u} -> {r.status_code} {len(r.text)}B final={r.url}")
            if r.status_code == 200 and r.text and len(r.text) > 200:
                return r
        except requests.RequestException as e:
            print(f"[HTTP] ERR {u}: {e}")
    return None

def save_debug_html(text: str, path="debug_editsoft_ro.html"):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[DEBUG] HTML salvat în {path}")
    except Exception as e:
        print(f"[DEBUG] Nu pot salva HTML: {e}")

def extract_links(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    # respectă <base href>
    base_tag = soup.find("base", href=True)
    if base_tag:
        try:
            base_url = urljoin(base_url, base_tag["href"])
        except Exception:
            pass
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        full = urljoin(base_url, href)
        links.append(full)
    # dedupe păstrând ordinea
    seen = set()
    dedup = []
    for l in links:
        if l not in seen:
            dedup.append(l)
            seen.add(l)
    return dedup

def try_sitemaps(root: str, timeout=15):
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    cands = [
        urljoin(root, "/sitemap.xml"),
        urljoin(root, "/sitemap_index.xml"),
        urljoin(root, "/sitemap-index.xml"),
        urljoin(root, "/sitemap1.xml"),
    ]
    out = []
    for u in cands:
        try:
            r = sess.get(u, timeout=timeout)
            print(f"[SITEMAP] {u} -> {r.status_code} {len(r.text)}B")
            if r.status_code == 200 and "<urlset" in r.text or "<sitemapindex" in r.text:
                # extragere URL-uri simple cu regex tolerant
                out += re.findall(r"<loc>\s*([^<]+)\s*</loc>", r.text, re.I)
        except requests.RequestException:
            pass
    # dedupe
    uniq = []
    seen = set()
    for u in out:
        u = u.strip()
        if u and u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq

def log_ignored(reason: str, url: str):
    print(f"[IGNORE] {reason}: {url}")

def filter_links(links, domain, ignore_prefixes=None, ignore_exact=None):
    ignore_prefixes = ignore_prefixes or []
    ignore_exact = set(ignore_exact or [])

    kept, ign = [], 0
    for l in links:
        if l in ignore_exact:
            log_ignored("exact", l); ign += 1; continue
        # doar pe același domeniu (fără subdomenii, ajustați dacă vreți *.editsoft.ro)
        host = (urlparse(l).netloc or "").lower()
        if host and host != domain and host != f"www.{domain}":
            log_ignored("alt host", l); ign += 1; continue
        hit = False
        for pref in ignore_prefixes:
            if l.startswith(pref):
                log_ignored(f"prefix {pref}", l); ign += 1; hit = True; break
        if hit:
            continue
        kept.append(l)
    print(f"[FILTER] total={len(links)} kept={len(kept)} ignored={ign}")
    return kept

def crawl_home(u: str, ignore_prefixes, ignore_exact):
    variants = canon_variants(u)
    r = fetch_first_ok(variants)
    if not r:
        print("[ERR] Nicio variantă nu a returnat conținut HTML utilizabil (verifică 403/JS).")
        return [], None
    save_debug_html(r.text)
    links = extract_links(r.text, r.url)

    domain = urlparse(r.url).netloc.lower()
    if domain.startswith("www."): domain = domain[4:]

    links = filter_links(links, domain, ignore_prefixes, ignore_exact)

    if not links:
        print("[INFO] Niciun <a> util. Încerc sitemap...")
        sm = try_sitemaps(f"https://{domain}/")
        if sm:
            print(f"[SITEMAP] găsit {len(sm)} URL-uri din sitemap.")
            # ținem doar URL-urile din același domeniu
            sm = filter_links(sm, domain, ignore_prefixes, ignore_exact)
            return sm, r.url
    return links, r.url

# EXEMPLE DE FOLOSIRE ÎN SCRIPTUL TĂU:
if __name__ == "__main__":
    site = input("Introdu URL-ul site-ului: ").strip()
    # TODO: încarcă ignore-urile tale reale
    ignore_prefixes = []  # ex: ["https://editsoft.ro/wp-json", ...]
    ignore_exact = []     # ex: ["https://editsoft.ro/"]
    links, final_url = crawl_home(site, ignore_prefixes, ignore_exact)
    print(f"[REZ] final_url={final_url} links={len(links)}")
    for l in links[:50]:
        print(" -", l)
    if not links:
        print("Nu am găsit linkuri nici după sitemap.")
