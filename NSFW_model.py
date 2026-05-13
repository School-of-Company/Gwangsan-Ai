from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
model_id = "Falconsai/nsfw_image_detection"
processor = ViTImageProcessor.from_pretrained(model_id)
model = ViTForImageClassification.from_pretrained(model_id)

image = Image.open(r"C:\Users\byb\Pictures\Screenshots\2026-04-06_011113.png").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

predicted_class = logits.argmax(-1).item()
label = model.config.id2label[predicted_class]

print(f"결과: {label}")

import torch.nn.functional as F
probs = F.softmax(logits, dim=-1)
for i, prob in enumerate(probs[0]):
    print(f"{model.config.id2label[i]}: {prob:.4f}")