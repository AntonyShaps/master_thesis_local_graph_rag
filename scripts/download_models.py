from paths import MODELS_DIR 
from huggingface_hub import snapshot_download
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    local_dir: str
    allow_patterns: list[str] | None = None

MODELS: list[ModelSpec] = [
        ModelSpec(
            repo_id = "nanonets/Nanonets-OCR2-3B",
            local_dir = MODELS_DIR / "Nanonets-OCR2-3B",
            ),
      ModelSpec(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            local_dir = MODELS_DIR / "Mistral-7B-Instruct-v0.3",
            allow_patterns=["params.json",
                            "consolidated.safetensors",
                            "tokenizer.model.v3"]
            )
        ]
 
def ensure_model_download(spec):
    if spec.local_dir.exists():
        print(f"{spec.repo_id} is already present at:\n{spec.local_dir}")
        return spec.local_dir
    spec.local_dir.mkdir(parents=True)
    print(f"Dowloading: {spec.repo_id} -> {spec.local_dir}")

    snapshot_download(
        repo_id=spec.repo_id,
        local_dir=spec.local_dir,
        allow_patterns=spec.allow_patterns
    )
    print("Done!")
    return spec.local_dir


                         
if __name__ == "__main__":
    for spec in MODELS:
        ensure_model_download(spec)

