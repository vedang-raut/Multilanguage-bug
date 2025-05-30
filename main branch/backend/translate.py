# backend/translate.py

from transformers import MarianMTModel, MarianTokenizer
from backend.config import language_pairs

def load_translation_model(pair):
    model_name = language_pairs.get(pair)
    if model_name:
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        return model, tokenizer
    return None, None

def translate_text(text, src_lang, tgt_lang):
    if src_lang == tgt_lang:
        return text

    pair = (src_lang, tgt_lang)
    model, tokenizer = load_translation_model(pair)
    if model and tokenizer:
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated_tokens = model.generate(**inputs)
        return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    # Fallback via English
    pivot = translate_text(text, src_lang, "en")
    return translate_text(pivot, "en", tgt_lang)
