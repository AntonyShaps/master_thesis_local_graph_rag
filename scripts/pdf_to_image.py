from pathlib import Path
from paths import DATA_IMGS_DIR, DATA_RAW_DIR
from tqdm import tqdm
import fitz

def pdf_to_images(pdf_path: Path, out_dir: Path):
    doc = fitz.open(pdf_path)
    out_dir.mkdir(parents = True, exist_ok = True)

        
    for i, page in enumerate(tqdm(doc, desc=f"Processing {pdf_path.stem}", unit="page")):
        pix = page.get_pixmap(dpi=200)
        pix.save(out_dir / f"page_{i + 1:03d}.png")

    print(f"\nFinished: {len(doc)} pages saved to {out_dir}")
    

if __name__ == "__main__":
    if not DATA_IMGS_DIR.exists():
        DATA_IMGS_DIR.mkdir(parents = True)
        print(f"Created directory: {DATA_IMGS_DIR}")
    else:
        print(f"Directory already exists: {DATA_IMGS_DIR}")
    for pdf_path in DATA_RAW_DIR.glob("*.pdf"):
        out_dir = DATA_IMGS_DIR / pdf_path.stem
        pdf_to_images(pdf_path, out_dir)
   





