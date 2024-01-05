# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from transformers import pipeline
import torch
import os


PROMPT_TEMPLATE = """\
<|im_start|>system
You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!
<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        device ="cuda" if torch.cuda.is_available() else "cpu"

        # list all files in the current directory
        files = os.listdir("/")
        print(files)

        pipe = pipeline("text-generation", model="/dolphin-2.6-mistral-7B-AWQ", device=device)
        self.model = pipe


    def predict(
        self,
        prompt: str = Input(description="Grayscale input image"),
        prompt_template: str = Input(
            default=PROMPT_TEMPLATE,
            description="Template for the prompt. Use <|im_start|> and <|im_end|> to mark the start and end of the prompt.",
        ),
        max_new_tokens: int = Input(
            default=DEFAULT_MAX_NEW_TOKENS,
            description="Maximum number of tokens to generate",
        ),
        temperature: float = Input(
            default=DEFAULT_TEMPERATURE,
            description="Controls randomness. Lower means more deterministic.",
        ),
        top_p: float = Input(
            default=DEFAULT_TOP_P,
            description="Controls diversity. Lower means more repetitive.",
        ),
        top_k: int = Input(
            default=DEFAULT_TOP_K,
            description="Controls diversity. Lower means more repetitive.",
        )
    ) -> str:
        
        prompt = prompt_template.format(prompt=prompt)
        output = self.model(
            prompt,
            max_length=len(prompt) + max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        return output[0]["generated_text"]
