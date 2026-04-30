import os
import random
import shutil

# Ścieżki do Twoich folderów
dataset_dir = 'dataset'
val_dir = 'val'
split_ratio = 0.2  # 10%

# Tworzymy folder walidacyjny, jeśli nie istnieje
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# Przechodzimy przez każdy podfolder w dataset (każdy budynek)
for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)

    # Sprawdzamy czy to na pewno folder
    if os.path.isdir(category_path):
        # Tworzymy odpowiadający podfolder w folderze 'val'
        val_category_path = os.path.join(val_dir, category)
        if not os.path.exists(val_category_path):
            os.makedirs(val_category_path)

        # Pobieramy listę wszystkich plików w danej kategorii
        images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

        # Obliczamy ile to 10%
        num_val = int(len(images) * split_ratio)

        # Losujemy pliki do przeniesienia
        val_images = random.sample(images, num_val)

        # Przenosimy pliki
        for img in val_images:
            src_path = os.path.join(category_path, img)
            dst_path = os.path.join(val_category_path, img)
            shutil.move(src_path, dst_path)

        print(f"Kategoria {category}: Przeniesiono {num_val} plików do walidacji.")

print("Gotowe! Podział zakończony pomyślnie.")