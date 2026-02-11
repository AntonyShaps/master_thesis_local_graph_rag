from pathlib import Path
import urllib.request
import urllib.error

from config import SOURCES

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"


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
    DATA_RAW_DIR.mkdir(parents = True, exist_ok = True)
    print(f"Created a dir if it has not existed yet")
    
    for name, url in SOURCES.items():
        out_file = DATA_RAW_DIR / f"{name}.pdf"
        if out_file.exists():
            print(f"Already exists: {name}")
        else:
            download_pdf(url, out_file)
            print(f"Downloaded: {name}")
    

if __name__ == "__main__":
    main()
