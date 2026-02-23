from pathlib import Path
import urllib.request
import urllib.error

from config import SOURCES
from paths import DATA_RAW_DIR

def download_pdf(url: str, out_path: Path):

    req = urllib.request.Request(
        url,
        headers = {"User-Agent": "Master-Thesis-ESG-knowledge-extraction"}
    )

    try:
        with urllib.request.urlopen(req) as resp:
            content_type = resp.headers.get("Content-Type","").lower()

            if "pdf" not in content_type:
                raise ValueError(f"Not a PDF: {url}")

            out_path.write_bytes(resp.read())

    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} for {url}") from e

    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error for {url}") from e

def main():
    if not DATA_RAW_DIR.exists():
        DATA_RAW_DIR.mkdir(parents = True)
        print(f"Created directory: {DATA_RAW_DIR}")
    else:
        print(f"Directory already exists: {DATA_RAW_DIR}")
        
    for name, url in SOURCES.items():
        out_file = DATA_RAW_DIR / f"{name}.pdf"
        if out_file.exists():
            print(f"Already exists: {name}")
        else:
            download_pdf(url, out_file)
            print(f"Downloaded: {name}")
    

if __name__ == "__main__":
    main()
