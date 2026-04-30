import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

class SimpleModel(nn.Module):
    def __init__(self, liczba_budynkow=10):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, liczba_budynkow)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(torch.relu(self.conv2(x)))  # 112 -> 56
        x = self.pool(torch.relu(self.conv3(x)))  # 56 -> 28
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

Model_src = 'stare_modele/model_pwr.pth'
val_dir = 'val'
batch_size = 16

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
if not os.path.exists(val_dir):
    print(f"BŁĄD: Nie znaleziono folderu {val_dir}!")
    exit()

val_data = datasets.ImageFolder(val_dir, transform=test_transforms)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
class_names = val_data.classes  # Automatycznie pobiera nazwy folderów

model = SimpleModel(liczba_budynkow=len(class_names))
if os.path.exists(Model_src):
    model.load_state_dict(torch.load(Model_src, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model {Model_src} załadowany pomyślnie.")
else:
    print(f"BŁĄD: Nie znaleziono pliku wag: {Model_src}")
    exit()


# 6. Funkcja generująca wyniki
def evaluate_and_plot(model, loader, classes):
    y_true = []
    y_pred = []

    print("Analizowanie danych...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # --- MACIERZ POMYŁEK ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Przewidziane (Predykcja)', fontsize=12)
    plt.ylabel('Prawdziwe (Etykieta)', fontsize=12)
    plt.title('Macierz Pomyłek - Rozpoznawanie Budynków PWr', fontsize=15)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Zapis wykresu do pliku (opcjonalnie do sprawozdania)
    plt.savefig('macierz_pomylek.png')
    print("Wykres został zapisany jako 'macierz_pomylek.png'")
    plt.show()

    # --- RAPORT KLASYFIKACJI ---
    print("\n" + "=" * 60)
    print("SZCZEGÓŁOWY RAPORT KLASYFIKACJI")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=classes))
    print("=" * 60)


# Uruchomienie testów
if __name__ == "__main__":
    evaluate_and_plot(model, val_loader, class_names)