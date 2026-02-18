# whisper is lazy-loaded
import os

_model = None

def get_whisper_model():
    global _model
    if _model is None:
        print("⏳ Loading Whisper model ('base')...")
        import whisper
        # Use 'tiny' for Render Free tier compatibility (512MB RAM limit)
        model_name = os.getenv('WHISPER_MODEL', 'tiny')
        _model = whisper.load_model(model_name)
    return _model

def clear_whisper_model():
    global _model
    _model = None
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
