from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, ElectraForSequenceClassification

MODEL_NAME = "kdyeon0309/gogo_forpanity_filter"
TOKENIZER_NAME = "beomi/KcELECTRA-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    yield
    del tokenizer, model


app = FastAPI(lifespan=lifespan)


class TextRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(req: TextRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    try:
        inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        return {"label": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
