from transformers import AutoTokenizer, ElectraForSequenceClassification
import torch

MODEL_NAME     = "kdyeon0309/gogo_forpanity_filter"
TOKENIZER_NAME = "beomi/KcELECTRA-base"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model     = ElectraForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

def is_profanity(text):
    return predict(text) == 1

texts = [
    "섹스",
    "정신병자",
    "좆물",
]

for text in texts:
    cls = predict(text)
    flag = "비속어" if is_profanity(text) else "정상"
    print(f"{flag} (class={cls})  {text}")