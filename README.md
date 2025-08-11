# 📄 Finance Document Classification API

A fine-tuned [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) model served via **FastAPI** for classifying finance-related documents. It uses a DistilBERT base model fine-tuned on the English subset of the Synthetic PII Finance Multilingual dataset. The API can be run locally or inside Docker and offers both `/predict` and `/health` endpoints.

## 🚀 Features
- Fine-tuned DistilBERT-based classification model
- REST API with `/predict` and `/health` endpoints
- Docker-ready for easy deployment
- High accuracy with production-ready code

## 📊 Model Details
- **Base Model:** distilbert-base-uncased  
- **Task:** Multi-class finance document classification  
- **Language:** English  
- **Dataset:** Synthetic PII Finance Multilingual (English subset)  
- **Framework:** Hugging Face Transformers  
- **Metrics:**  
  | Metric      | Score   |
  |-------------|---------|
  | Accuracy    | 98.65%  |
  | Precision   | 98.70%  |
  | Recall      | 98.65%  |
  | F1          | 98.65%  |

## 📂 Project Structure
```
finance_document_classification/
├─ app/
│  └─ main.py               # FastAPI app
├─ final_model/              # Saved model & tokenizer
├─ requirements.txt
├─ Dockerfile
├─ .dockerignore
└─ README.md
```

## 🛠 Installation
Clone the repository:
```bash
git clone https://github.com/Ar86Bat/Finance-Document-Text-Classification.git
cd Finance-Document-Text-Classification
```

Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## ▶️ Run Locally
```bash
uvicorn app.main:app --reload
```
The API will be available at:
```
http://127.0.0.1:8000/docs
```

## 🐳 Run with Docker
```bash
docker build -t finance-doc-classifier .
docker run -p 8000:8000 finance-doc-classifier
```

## 📡 API Endpoints
### `POST /predict`
**Request:**
```json
{
  "text": "Client requested details about investment restrictions."
}
```
**Response:**
```json
{
  "label": "Investment Restrictions",
  "confidence": 0.987
}
```

### `GET /health`
Returns API health status.

## 📦 Use the Model in Python
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "Ar86Bat/Finance-Document-Text-Classification"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

text = "Client requested details about investment restrictions."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_id = torch.argmax(probs, dim=1).item()

print("Predicted class ID:", pred_id)
```

## 📜 License
MIT License.
