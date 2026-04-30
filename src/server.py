from fastapi import FastAPI, UploadFile, File
import uvicorn
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

CONFIDENCE_THRESHOLD = 60.0

# struktura sieci
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))


        x = x.view(x.size(0), -1)  # Spłaszczanie
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ladowanie wag
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleModel()
model.load_state_dict(torch.load("model_pwr_best.pth", map_location=device, weights_only=True))
model.to(device)
model.eval() # wyłącza nauke

# transformacje
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# slownik nazw
class_names = ["A-1", "C-1", "C-13 (Serowiec)", "C-16", "C-18", "C-3/4", "C-7", "D-1", "D-20", "H-6"]

# endpoint przyjmujacy zdjecia
@app.post("/predict")
async def predict_building(file: UploadFile = File(...)):
    try:
        # odczyt z telefonu
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        #przygotowanie i wyslanie do modelu
        input_tensor = val_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            conf, predicted = torch.max(probabilities, 1)

        confidence = conf.item() * 100
        building_idx = predicted.item()
        building = class_names[building_idx]
        print(f"Wykryty budynek: {building} ({confidence:.2f}%)")
        if(confidence < CONFIDENCE_THRESHOLD):
            return {
                "budynek": "Nie rozpoznano",
                "pewnosc": round(confidence, 2),
                "info": "Pewność poniżej progu"
            }
        else:
            return {
                "budynek": building,
                "pewnosc": round(confidence, 2),
                "info": "OK"
            }
    except Exception as e:
        return {"error": f"Błąd przetwarzania obrazu: {str(e)}"}

if __name__ == "__main__":
    print("Starting server PWr Scanner...")
    uvicorn.run(app, host="0.0.0.0", port=8000)