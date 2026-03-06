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


def ocr_page_with_nanonets_s(image, model, processor, max_new_tokens=15000):
    prompt = """Extract all readable text from this page in reading order.
Output as markdown with these rules:
1) Tables: use raw HTML <table>...</table>.
2) Equations: use LaTeX between $$...$$ for block equations and $...$ for inline equations.
3) Figures/images: output one <img>...</img> tag per figure in reading order.
   - If caption exists, put the exact caption text inside the tag.
   - If no caption exists, write a concise one-sentence description.
4) Page number: include exactly once as <page_number>N</page_number> if visible.
5) If text is unreadable, write [illegible] at that location.
Do not add explanations or metadata beyond the extracted content."""
    messages = [
        {
            "role": "system",
            "content": "You are an OCR extraction assistant. Follow the output rules exactly and preserve content order.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"{image}"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text[0]


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

        with md_file.open("w") as f:
            for image_path in sorted(company_dir.iterdir()):
                print(f"  OCR: {image_path.name}")
                image = Image.open(image_path)
                result = ocr_page_with_nanonets_s(image, model, processor)
                f.write(result)
