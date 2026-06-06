from contextlib import asynccontextmanager
from io import BytesIO
import asyncio

import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_ID = "AdamCodd/vit-base-nsfw-detector"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
NSFW_THRESHOLD = 0.35

processor = None
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    model = ViTForImageClassification.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    yield
    processor = None
    model = None


app = FastAPI(lifespan=lifespan)


def _infer(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.cpu()


@app.post("/nsfw")
async def predict(file: UploadFile = File(...)):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image_bytes = await file.read()
        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="Image too large (max 10MB)")
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        loop = asyncio.get_running_loop()
        logits = await loop.run_in_executor(None, _infer, image)
        probs = F.softmax(logits, dim=-1)
        probabilities = {
            model.config.id2label[i]: round(prob.item(), 4)
            for i, prob in enumerate(probs[0])
        }
        nsfw_score = next(
            v for k, v in probabilities.items() if k.lower() == "nsfw"
        )
        return {"is_nsfw": nsfw_score >= NSFW_THRESHOLD}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
