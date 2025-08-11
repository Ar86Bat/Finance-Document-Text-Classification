from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, pickle

MODEL_DIR = "Ar86Bat/Finance-Document-Text-Classification"

app = FastAPI(title="Finance Doc Classifier")

# Load once at startup
@app.on_event("startup")
def load_artifacts():
    global tokenizer, model, label_encoder, device
    with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    device = torch.device("cpu")  # change if you have GPU
    model.to(device)

class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    label: str
    confidence: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    inputs = tokenizer(
        payload.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0][pred_id].item())

    label = label_encoder.inverse_transform([pred_id])[0]
    return {"label": label, "confidence": confidence}
