from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "MNIST_CNN_PYTORCH.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# IMPORTANT: This architecture must exactly match the CNN used to create
# MNIST_CNN_PYTORCH.pth. Replace this class if your training architecture differs.
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"{MODEL_PATH} not found. Put MNIST_CNN_PYTORCH.pth beside app.py."
    )

model = MNISTCNN().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

app = FastAPI(
    title="MNIST CNN PyTorch API",
    description="Handwritten digit prediction using a trained PyTorch CNN.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {"message": "MNIST CNN PyTorch API is running", "device": str(DEVICE)}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        data = await file.read()
        image = Image.open(BytesIO(data)).convert("L")
        x = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

        digit = int(predicted.item())
        conf = float(confidence.item())

        return {
            "predicted_digit": digit,
            "confidence": round(conf, 4),
            "confidence_percent": round(conf * 100, 2),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")
