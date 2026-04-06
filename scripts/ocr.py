from paths import DATA_IMGS_DIR, DATA_MARKDOWN_DIR, MODELS_DIR
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

model_path = MODELS_DIR / "Nanonets-OCR2-3B"
model = AutoModelForImageTextToText.from_pretrained(
    model_path, dtype="auto", device_map="auto", attn_implementation="flash_attention_2"
)

model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)
def ocr_page_with_nanonets_s(image, model, processor, max_new_tokens=5000):
    prompt = """
Extract all readable content from this page in strict reading order.

Output rules:
1) Preserve headings as markdown headings (#, ##, ###) whenever visually clear.
2) Preserve paragraphs as normal markdown text.
3) Preserve bullet lists and numbered lists as markdown lists.
4) Tables: output as raw HTML <table>...</table>. Reconstruct rows and columns as faithfully as possible.
5) Figures/charts/images:
   - If a visible caption exists, output exactly:
     <figure>EXACT_CAPTION_TEXT</figure>
   - If there is no visible caption, output exactly:
     <figure></figure>
6) Equations:
   - block equations as $$...$$
   - inline equations as $...$
7) If text is unreadable, write [illegible] at that location.
8) Do not summarize, explain, correct, or normalize the content.
9) Preserve numbers, units, percentages, years, and terminology exactly as written.
10) Do not output page-number tags or any extra metadata.

Return only the extracted page content.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a layout-aware OCR extraction assistant. "
                "Extract only what is visibly present on the page. "
                "Preserve reading order and structure exactly. "
                "Do not summarize, infer, or describe content beyond the requested format."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return output_text[0].strip()


if __name__ == "__main__":
    if not DATA_MARKDOWN_DIR.exists():
        DATA_MARKDOWN_DIR.mkdir(parents=True)
        print(f"Created directory: {DATA_MARKDOWN_DIR}")
    else:
        print(f"Directory already exists: {DATA_MARKDOWN_DIR}")

    for company_dir in DATA_IMGS_DIR.iterdir():
        company_name = company_dir.name
        md_file = DATA_MARKDOWN_DIR / f"{company_name}.md"

        print(f"\nProcessing: {company_name}")
        print(f"Markdown file: {md_file}")

        with md_file.open("w", encoding="utf-8") as f:
            f.write(f"# {company_name}\n\n")
        
            for page_idx, image_path in enumerate(sorted(company_dir.iterdir()), start=1):
                print(f"  OCR: {image_path.name}")
                image = Image.open(image_path).convert("RGB")
                result = ocr_page_with_nanonets_s(image, model, processor)
        
                f.write(f"\n\n--- PAGE {page_idx} START ---\n\n")
                f.write(result.strip())
                f.write(f"\n\n--- PAGE {page_idx} END ---\n\n")
