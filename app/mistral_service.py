from __future__ import annotations

from pathlib import Path
from threading import Lock

from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.generate import generate
from mistral_inference.transformer import Transformer


class MistralService:
    def __init__(
        self,
        model_dir: Path,
        default_max_new_tokens: int = 512,
        default_temperature: float = 0.1,
    ) -> None:
        self._model_dir = model_dir
        self._default_max_new_tokens = default_max_new_tokens
        self._default_temperature = default_temperature

        self._model: Transformer | None = None
        self._tokenizer: MistralTokenizer | None = None
        self._eos_id: int | None = None
        self._load_lock = Lock()
        self._generate_lock = Lock()

    @property
    def model_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _ensure_loaded(self) -> None:
        if self.model_loaded:
            return

        with self._load_lock:
            if self.model_loaded:
                return

            if not self._model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {self._model_dir}")

            tokenizer_path = self._model_dir / "tokenizer.model.v3"
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

            self._tokenizer = MistralTokenizer.from_file(str(tokenizer_path))
            self._model = Transformer.from_folder(str(self._model_dir))
            self._eos_id = self._tokenizer.instruct_tokenizer.tokenizer.eos_id

    def generate_answer(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self._ensure_loaded()

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._eos_id is not None

        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompt)]
        )
        tokens = self._tokenizer.encode_chat_completion(completion_request).tokens

        with self._generate_lock:
            output_tokens, _ = generate(
                [tokens],
                self._model,
                max_tokens=max_new_tokens or self._default_max_new_tokens,
                temperature=(
                    self._default_temperature if temperature is None else temperature
                ),
                eos_id=self._eos_id,
            )

        decoded = self._tokenizer.instruct_tokenizer.tokenizer.decode(output_tokens[0])
        return decoded.strip()
