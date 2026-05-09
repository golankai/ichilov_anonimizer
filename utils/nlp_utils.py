import nltk
import torch
from typing import Any

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def setup_nltk() -> None:
    """
    Ensure all required NLTK components are downloaded and available.

    Currently checks for:
    - 'punkt'
    - 'punkt_tab'
    """
    for package in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{package}")
        except (LookupError, OSError):
            print(f"--- Downloading NLTK resource: {package} ---")
            nltk.download(package, quiet=True)

# factory function returning nltk sentence tokenizer
def get_sentence_tokenizer(tokenizer_type: str) -> Any:
    if tokenizer_type == 'nltk':
        return nltk.sent_tokenize
    elif tokenizer_type == 'spacy':
        import spacy
        return spacy.load('en_core_web_sm').tokenizer
    else:
        raise ValueError(f"Invalid tokenizer type: {tokenizer_type}")
    """
    Detect hardware and initialize the Hugging Face NER pipeline.

    Args:
        model_path: Path or identifier of the trained model directory.

    Returns:
        The initialized Hugging Face token classification pipeline.
    """


def get_pipeline(model_path: str) -> Any:
    """
    Detect hardware and initialize the Hugging Face NER pipeline.

    Args:
        model_path: Path or identifier of the trained model directory.

    Returns:
        The initialized Hugging Face token classification pipeline.
    """
    device: Any
    if torch.cuda.is_available():
        device = 0
        print("--- Using GPU (CUDA) ---")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("--- Using Apple Silicon GPU (MPS) ---")
    else:
        device = -1
        print("--- Using CPU ---")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )

