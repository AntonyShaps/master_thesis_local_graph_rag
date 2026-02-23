from paths import MODELS_DIR 
from huggingface_hub import snapshot_download


nanonets_model = "nanonets/Nanonets-OCR2-3B"
nanonets_dir = MODELS_DIR / "Nanonets-OCR2-3B"

mistral_model = "mistralai/Mistral-7B-v0.3"
mistral_dir = MODELS_DIR / "Mistral-7B-v0.3"

if __name__ == "__main__":
    if not nanonets_dir.exists() and not mistral_dir.exists():
        nanonets_dir.mkdir(parents=True)
        mistral_dir.mkdir(parents=True)
        print(f"Created directories: \n{nanonets_dir}\n{mistral_dir}")
    else:
        print("Already there")

    snapshot_download(repo_id=mistral_model,
                      allow_patterns=["params.json",
                                      "consolidated.safetensors",
                                      "tokenizer.model.v3"],
                      local_dir=mistral_dir
                      )
    snapshot_download(repo_id=nanonets_model,
                      local_dir=nanonets_dir
                      )
 
