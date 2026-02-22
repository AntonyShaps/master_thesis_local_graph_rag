from paths import DATA_IMGS_DIR, DATA_MARKDOWN_DIR
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import json
model_path = "nanonets/Nanonets-OCR2-3B"
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype = "auto",
    device_map = "auto",
    attn_implementation = "flash_attention_2"
)

model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)
JSON_PROMPT_TEMPLATE = """
Task:
Extract all readable content from the image of a single report page.

Rules:
- Do NOT invent text. If something is not legible, use null and add a short note in "warnings".
- Keep reading order: top-to-bottom, left-to-right.
- Separate repetitive page furniture:
  - Put running headers into blocks with type="header"
  - Put footers/page numbers/legal lines into blocks with type="footer"
- Preserve section hierarchy when obvious using "section_path" (list of headings encountered so far).
- Tables:
  - If you detect a table, output one block with type="table"
  - Provide "html" AND a normalized "rows" 2D array
  - Keep units in headers/cells
- Figures:
  - If there is an image/figure, create a block type="figure"
  - If caption exists, copy it exactly into "caption"
 Checkboxes: use "☑" and "☐" in text.
"""

def ocr_page_nanonets(
    image: Image.Image,
    model,
    processor,
    max_new_tokens: int = 8192,
):

    prompt = JSON_PROMPT_TEMPLATE

    messages = [
        {"role": "system", "content": "You extract text from documents."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]

    chat = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(text=chat, images=image, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=1.0,
            repetition_penalty=1.05,   # mild, helps repeated headers sometimes
        )

    generated = output_ids[:, inputs.input_ids.shape[1]:]
    text_output = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0].strip()
    return text_output 

if __name__ == "__main__":
    if not DATA_MARKDOWN_DIR.exists():
        DATA_MARKDOWN_DIR.mkdir(parents = True)
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
                result = ocr_page_nanonets(image,model,processor)
                f.write(result)
