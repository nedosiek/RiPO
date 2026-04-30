import torch

if torch.cuda.is_available():
    # Pobiera całkowity VRAM dla domyślnej karty (indeks 0) i zamienia na GB
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Karta graficzna: {torch.cuda.get_device_name(0)}")
    print(f"Całkowity VRAM: {vram_total:.2f} GB")
else:
    print("CUDA nie jest dostępne.")