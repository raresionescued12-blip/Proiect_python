import json
from urllib.parse import urlparse
import mysql.connector

DB = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "webscrapping",
}

Tabel = {
    "site_uri": "site_uri",
    "tip_pagina": "tip_pagina",
    "domeniu": "domeniu",
    "categorie": "categorie",
    "subcategorie": "subcategorie",
    "meniuri_linkuri": "meniuri_linkuri",
    "meniuri_tip_pagina": "meniuri_tip_pagina",
    "linkuri_domenii": "linkuri_domenii",
    "linkuri_categorii": "linkuri_categorii",
    "linkuri_subcategorii": "linkuri_subcategorii",
    "site_uri_email": "site_uri_email",
    "site_uri_telefoane": "site_uri_telefoane",
    "site_uri_social_media": "site_uri_social_media",
    "site_uri_adresa_fizica": "site_uri_adresa_fizica",
    "produse": "produse",
}


def base_site(url: str) -> str:
    p = urlparse(url)
    host = p.netloc or url
    if host.startswith("www."):
        host = host[4:]
    return f"https://{host}"


def get_one(cur, sql, params):
    cur.execute(sql, params)
    return cur.fetchone()


def get_or_create_site(conn, cur, adresa: str) -> int:
    t = Tabel["site_uri"]
    r = get_one(cur, f"SELECT id_site FROM `{t}` WHERE adresa=%s LIMIT 1", (adresa,))
    if r:
        return r[0]
    cur.execute(f"INSERT INTO `{t}`(adresa) VALUES (%s)", (adresa,))
    conn.commit()
    return cur.lastrowid


def get_or_create_tip(conn, cur, den: str) -> int:
    t = Tabel["tip_pagina"]
    den = (den or "").strip() or "necunoscut"
    r = get_one(cur, f"SELECT id_tip_pagina FROM `{t}` WHERE denumirea_tipului=%s LIMIT 1", (den,))
    if r:
        return r[0]
    cur.execute(f"INSERT INTO `{t}`(denumirea_tipului) VALUES (%s)", (den,))
    conn.commit()
    return cur.lastrowid


def get_or_create_domeniu(conn, cur, den: str) -> int:
    t = Tabel["domeniu"]
    den = (den or "").strip() or "necunoscut"
    r = get_one(cur, f"SELECT id_domeniu FROM `{t}` WHERE denumire_domeniu=%s LIMIT 1", (den,))
    if r:
        return r[0]
    cur.execute(f"INSERT INTO `{t}`(denumire_domeniu) VALUES (%s)", (den,))
    conn.commit()
    return cur.lastrowid


def get_or_create_categorie(conn, cur, id_domeniu: int, den: str) -> int:
    t = Tabel["categorie"]
    den = (den or "").strip() or "General"
    r = get_one(cur, f"SELECT id_categorie FROM `{t}` WHERE id_domeniu=%s AND denumirea_categoriei=%s LIMIT 1",
                (id_domeniu, den))
    if r:
        return r[0]
    cur.execute(f"INSERT INTO `{t}`(id_domeniu, denumirea_categoriei) VALUES (%s,%s)", (id_domeniu, den))
    conn.commit()
    return cur.lastrowid


def get_or_create_subcategorie(conn, cur, id_categorie: int, den: str, id_link: int | None = None) -> int:
    t = Tabel["subcategorie"]
    den = (den or "").strip() or "General"
    if id_link is not None:
        r = get_one(cur, f"SELECT id_subcategorie FROM `{t}` WHERE id_categorie=%s AND denumirea_subcategoriei=%s AND id_link=%s LIMIT 1",
                    (id_categorie, den, id_link))
        if r:
            return r[0]
        cur.execute(f"INSERT INTO `{t}`(id_categorie, denumirea_subcategoriei, id_link) VALUES (%s,%s,%s)",
                    (id_categorie, den, id_link))
    else:
        r = get_one(cur, f"SELECT id_subcategorie FROM `{t}` WHERE id_categorie=%s AND denumirea_subcategoriei=%s LIMIT 1",
                    (id_categorie, den))
        if r:
            return r[0]
        cur.execute(f"INSERT INTO `{t}`(id_categorie, denumirea_subcategoriei) VALUES (%s,%s)",
                    (id_categorie, den))
    conn.commit()
    return cur.lastrowid


def get_or_create_link(conn, cur, id_site: int, url: str) -> int:
    t = Tabel["meniuri_linkuri"]
    r = get_one(cur, f"SELECT id_link FROM `{t}` WHERE id_site=%s AND url=%s LIMIT 1", (id_site, url))
    if r:
        return r[0]
    cur.execute(f"INSERT INTO `{t}`(id_site, url) VALUES (%s,%s)", (id_site, url))
    conn.commit()
    return cur.lastrowid


def ensure_meniu_tip(conn, cur, id_link: int, id_tip: int):
    t = Tabel["meniuri_tip_pagina"]
    r = get_one(cur, f"SELECT 1 FROM `{t}` WHERE id_link=%s AND id_tip_pagina=%s LIMIT 1", (id_link, id_tip))
    if r:
        return
    cur.execute(f"INSERT INTO `{t}`(id_link, id_tip_pagina) VALUES (%s,%s)", (id_link, id_tip))
    conn.commit()


def ensure_link_dom(conn, cur, id_link: int, id_domeniu: int):
    t = Tabel["linkuri_domenii"]
    r = get_one(cur, f"SELECT 1 FROM `{t}` WHERE id_link=%s AND id_domeniu=%s LIMIT 1", (id_link, id_domeniu))
    if r:
        return
    cur.execute(f"INSERT INTO `{t}`(id_link, id_domeniu) VALUES (%s,%s)", (id_link, id_domeniu))
    conn.commit()


def ensure_link_cat(conn, cur, id_link: int, id_categorie: int):
    t = Tabel["linkuri_categorii"]
    r = get_one(cur, f"SELECT 1 FROM `{t}` WHERE id_link=%s AND id_categorie=%s LIMIT 1", (id_link, id_categorie))
    if r:
        return
    cur.execute(f"INSERT INTO `{t}`(id_link, id_categorie) VALUES (%s,%s)", (id_link, id_categorie))
    conn.commit()


def ensure_link_subcat(conn, cur, id_link: int, id_categorie: int, id_subcategorie: int):
    t = Tabel["linkuri_subcategorii"]
    cur.execute(f"SHOW COLUMNS FROM `{t}` LIKE %s", ("id_subcategorie",))
    if cur.fetchone() is None:
        raise RuntimeError(f"`{t}` nu are coloana `id_subcategorie`. Adauga coloana in schema.")
    r = get_one(cur, f"SELECT 1 FROM `{t}` WHERE id_link=%s AND id_subcategorie=%s LIMIT 1",
                (id_link, id_subcategorie))
    if r:
        return
    cur.execute(f"INSERT INTO `{t}`(id_link, id_subcategorie) VALUES (%s,%s)", (id_link, id_subcategorie))
    conn.commit()


def insert_email(conn, cur, id_site: int, email: str):
    t = Tabel["site_uri_email"]
    email = (email or "").strip()
    if not email:
        return
    r = get_one(cur, f"SELECT 1 FROM `{t}` WHERE id_site=%s AND email=%s LIMIT 1", (id_site, email))
    if r:
        return
    cur.execute(f"INSERT INTO `{t}`(id_site, email) VALUES (%s,%s)", (id_site, email))
    conn.commit()


def insert_telefon(conn, cur, id_site: int, tel: str):
    t = Tabel["site_uri_telefoane"]
    tel = (tel or "").strip()
    if not tel:
        return
    r = get_one(cur, f"SELECT 1 FROM `{t}` WHERE id_site=%s AND telefon=%s LIMIT 1", (id_site, tel))
    if r:
        return
    cur.execute(f"INSERT INTO `{t}`(id_site, telefon) VALUES (%s,%s)", (id_site, tel))
    conn.commit()


def insert_social(conn, cur, id_site: int, link: str):
    t = Tabel["site_uri_social_media"]
    link = (link or "").strip()
    if not link:
        return
    r = get_one(cur, f"SELECT 1 FROM `{t}` WHERE id_site=%s AND link=%s LIMIT 1", (id_site, link))
    if r:
        return
    cur.execute(f"INSERT INTO `{t}`(id_site, link) VALUES (%s,%s)", (id_site, link))
    conn.commit()


def insert_adresa(conn, cur, id_site: int, adresa_text: str):
    t = Tabel["site_uri_adresa_fizica"]
    adresa_text = (adresa_text or "").strip()
    if not adresa_text:
        return
    r = get_one(cur, f"SELECT 1 FROM `{t}` WHERE id_site=%s AND strada=%s LIMIT 1", (id_site, adresa_text))
    if r:
        return
    cur.execute(f"INSERT INTO `{t}`(id_site, strada) VALUES (%s,%s)", (id_site, adresa_text))
    conn.commit()


def upsert_produs(conn, cur, id_link: int, url: str, categorie: str):
    t = Tabel["produse"]
    url = (url or "").strip()
    categorie = (categorie or "").strip() or "necunoscut"
    if not url:
        return
    r = get_one(cur, f"SELECT id_produs, categorie FROM `{t}` WHERE id_link=%s AND url=%s LIMIT 1",
                (id_link, url))
    if r:
        id_produs, old_cat = r
        old_cat = (old_cat or "").strip()
        if old_cat != categorie:
            cur.execute(f"UPDATE `{t}` SET categorie=%s WHERE id_produs=%s", (categorie, id_produs))
            conn.commit()
        return
    cur.execute(f"INSERT INTO `{t}`(id_link, url, categorie) VALUES (%s,%s,%s)", (id_link, url, categorie))
    conn.commit()


def main():
    with open("REZULTATE_FINALE.json", encoding="utf-8") as f:
        data = json.load(f)

    conn = mysql.connector.connect(**DB)
    cur = conn.cursor()

    # determinăm site-ul de bază
    if data.get("pagini_clasificate"):
        site_val = data["pagini_clasificate"][0]["site"]
    elif data.get("servicii", {}).get("pages"):
        site_val = data["servicii"]["pages"][0]["url"]
    elif data.get("contact"):
        site_val = data["contact"][0]["site"]
    else:
        raise RuntimeError("JSON fara pagini_clasificate/servicii/contact.")

    id_site = get_or_create_site(conn, cur, base_site(site_val))

    # pagini clasificate
    for p in data.get("pagini_clasificate", []):
        page_type = (p.get("categorie") or "necunoscut").strip().lower()
        url = p["url"]
        id_link = get_or_create_link(conn, cur, id_site, url)
        id_dom = get_or_create_domeniu(conn, cur, page_type)
        ensure_link_dom(conn, cur, id_link, id_dom)
        id_tip_page = get_or_create_tip(conn, cur, page_type)
        ensure_meniu_tip(conn, cur, id_link, id_tip_page)

    # servicii (pages din JSON)
    for s in data.get("servicii", {}).get("pages", []):
        url = s["url"]
        page_type = (s.get("categorie_originala") or "servicii").strip().lower()
        cat_name = (s.get("categorie_prezisa") or "").strip()
        subcat_name = (s.get("subcategorie_prezisa") or "").strip()

        id_link = get_or_create_link(conn, cur, id_site, url)
        id_dom = get_or_create_domeniu(conn, cur, page_type)
        ensure_link_dom(conn, cur, id_link, id_dom)

        if cat_name:
            id_cat = get_or_create_categorie(conn, cur, id_dom, cat_name)
            ensure_link_cat(conn, cur, id_link, id_cat)
            if subcat_name:
                id_sub = get_or_create_subcategorie(conn, cur, id_cat, subcat_name)
                ensure_link_subcat(conn, cur, id_link, id_cat, id_sub)

        id_tip_page = get_or_create_tip(conn, cur, page_type)
        ensure_meniu_tip(conn, cur, id_link, id_tip_page)
        if cat_name:
            id_tip_cat = get_or_create_tip(conn, cur, cat_name)
            ensure_meniu_tip(conn, cur, id_link, id_tip_cat)

    # produse
    for p in data.get("produse", []):
        url = (p.get("url") or "").strip()
        if not url:
            continue
        categorie = (
            p.get("tip_produs")
            or p.get("categorie_finala")
            or p.get("categorie")
            or "necunoscut"
        ).strip()
        id_link = get_or_create_link(conn, cur, id_site, url)
        upsert_produs(conn, cur, id_link, url, categorie)

    # contacte
    for c in data.get("contact", []):
        url = c.get("url") or base_site(c.get("site", ""))
        id_link = get_or_create_link(conn, cur, id_site, url)

        id_dom = get_or_create_domeniu(conn, cur, "contact")
        ensure_link_dom(conn, cur, id_link, id_dom)

        id_tip_page = get_or_create_tip(conn, cur, "contact")
        ensure_meniu_tip(conn, cur, id_link, id_tip_page)

        for email in c.get("email", []):
            insert_email(conn, cur, id_site, email)
        for tel in c.get("telefon", []):
            insert_telefon(conn, cur, id_site, tel)
        for sm in c.get("social_media", []):
            insert_social(conn, cur, id_site, sm)
        if c.get("adresa_fizica"):
            insert_adresa(conn, cur, id_site, c["adresa_fizica"])

    cur.close()
    conn.close()
    print("Import reusit (idempotent + update categorie produse).")


if __name__ == "__main__":
    main()
