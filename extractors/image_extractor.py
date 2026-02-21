import os
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from extractors.pdf_extractor import download_file

load_dotenv()

# Global model cache
_blip_processor = None
_blip_model = None

def get_blip():
    """Lazy load BLIP model for image captioning/QA"""
    global _blip_processor, _blip_model
    if _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("⏳ Loading BLIP Image Vision model...")
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return _blip_processor, _blip_model

def extract_image(file_url):
    """
    Extracts description and potential text from an image.
    Uses BLIP for visual understanding and (optionally) EasyOCR if available.
    """
    try:
        # Download Image
        img_content = download_file(file_url)
        raw_image = Image.open(img_content).convert('RGB')
        
        # 1. Visual Description using BLIP
        processor, model = get_blip()
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
        
        print(f"🖼️ Image Vision Description: {description}")
        
        # 2. Text Extraction (OCR Fallback)
        text_content = ""
        try:
            # We check if EasyOCR is available (User can add it to requirements)
            import easyocr
            import numpy as np
            reader = easyocr.Reader(['en'])
            results = reader.readtext(np.array(raw_image))
            text_content = " ".join([res[1] for res in results])
            if text_content:
                print(f"📝 OCR Success: Found {len(text_content)} characters")
        except ImportError:
            # If no OCR library, description is our best bet for KG search
            print("ℹ️ EasyOCR not installed. Using visual description only.")
            text_content = description

        return {
            "text": text_content or description,
            "description": description,
            "summary": description,
            "thumbnail_url": file_url, # Already an image
            "metadata": {
                "extracted_at": "Now",
                "method": "BLIP-Vision + OCR-Fallback"
            }
        }
    except Exception as e:
        print(f"Error in extract_image: {str(e)}")
        raise e
