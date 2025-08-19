import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


HERE = Path(__file__).resolve().parent

F_REZULTATE_LLAMACOPIE = HERE / "rezultate_llama_copie.json"
F_PRODUSE_TIPURI       = HERE / "produse_tipuri.json"
F_SERVICII_SUBCAT      = HERE / "servicii_subcategorii.json"
F_CONTACTE_EXTRASE     = HERE / "contacte_extrase.json"

def ruleaza_scriptul(nume_script: str, url: str):
    print(f"[RUN] {nume_script} {url}")
    subprocess.run([sys.executable, str(HERE / nume_script), url], check=True)

def incarca_json(path: Path, fallback=None):
    if fallback is None:
        fallback = []
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return fallback

def ruleaza_inserare_date(cale_json: Path):
  
    print(f"[RUN] inserare_date.py (JSON: {cale_json})")
   
    subprocess.run([sys.executable, str(HERE / "inserare_date.py")], check=True)

def main():
    ap = argparse.ArgumentParser(description="Rulare si combinare rezultate.")
    ap.add_argument("url", help="URL-ul site-ului de analizat")
    ap.add_argument("-o", "--out", default=str(HERE / "REZULTATE_FINALE.json"), help="Calea fis final JSON")
    ap.add_argument("--print", action="store_true", help="Afisare json finanl.")
    ap.add_argument("--no-db", action="store_true", help="Nu rula inserarea in baza de date.")
    args = ap.parse_args()

    url = args.url

    to_run = [
        "analizare_site_llama.py",
        "clasificare_produse.py",
        "clasificare_site_v2.py",
        "date_contact.py",
    ]

    for script_name in to_run:
        ruleaza_scriptul(script_name, url)

    pagini_clasificate    = incarca_json(F_REZULTATE_LLAMACOPIE, [])
    produse_tipuri        = incarca_json(F_PRODUSE_TIPURI, [])
    servicii_subcategorii = incarca_json(F_SERVICII_SUBCAT, [])
    contacte_extrase      = incarca_json(F_CONTACTE_EXTRASE, [])

    rezultat_final = {
        "pagini_clasificate": pagini_clasificate,
        "produse": produse_tipuri,
        "servicii": servicii_subcategorii,
        "contact": contacte_extrase,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rezultat_final, f, ensure_ascii=False, indent=2)

    print(f"[OK] Rezultatul combinat a fost salvat in: {out_path}")
    if args.print:
        print(json.dumps(rezultat_final, ensure_ascii=False, indent=2))

    if not args.no_db:
        ruleaza_inserare_date(out_path)

if __name__ == "__main__":
    main()
