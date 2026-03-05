from paths import DATA_IMGS_DIR, DATA_MARKDOWN_DIR, MODELS_DIR
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
model_path = MODELS_DIR / "Nanonets-OCR2-3B"
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype = "auto",
    device_map = "auto",
    attn_implementation = "flash_attention_2"
)

model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

def ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=4096):
    prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number>."""
    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"{image_path}"},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]
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
                result = ocr_page_with_nanonets_s(image,model,processor)
                f.write(result)
