from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

def get_rendered_html(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # headless=True = fără fereastră vizibilă
        page = browser.new_page()
        page.goto(url, timeout=30000)  # 30 secunde timeout
        page.wait_for_timeout(3000)  # așteaptă 3 secunde pentru JS
        html = page.content()
        browser.close()
        return html

def extract_links(html):
    soup = BeautifulSoup(html, "html.parser")
    links = [a.get("href") for a in soup.find_all("a", href=True)]
    return links

if __name__ == "__main__":
    url = input("Introdu URL-ul site-ului: ").strip()
    html = get_rendered_html(url)
    links = extract_links(html)

    print(f"\nTotal linkuri găsite: {len(links)}")
    print("Primele 10 linkuri:")
    for link in links[:10]:
        print(link)
