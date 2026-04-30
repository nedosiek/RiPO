import cv2
import os

#folder z filmikami
path_video = "video/H6"
building_name = "H6"

folder_dataset = os.path.join('new_photos/dataset', building_name)
folder_val = os.path.join('new_photos/val', building_name)

#jesli nie istnieja to utworzyc
os.makedirs(folder_dataset, exist_ok=True)
os.makedirs(folder_val, exist_ok=True)

#szukanie wszystkich plikow wideo
formats = ('.mov', '.mp4', '.heic')
video_files = [f for f in os.listdir(path_video) if f.lower().endswith(formats)]

if not video_files:
    print(f"No video files found in {path_video}")
else:
    print(f"Found {len(video_files)} videos in {path_video}\n")

    saved_dataset = 0
    saved_val = 0

    for file in video_files:
        full_path = os.path.join(path_video, file)
        print(f"Processing {file}")

        cap = cv2.VideoCapture(full_path)
        fps_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break #koniec filmu

            # bierzmy co 20-a klatke (3 klatki na sekunde przy 60 fps)
            if fps_id % 20 == 0:
                number_of_photos = saved_dataset + saved_val

                # podzial na dataset/val = 80/20%
                if number_of_photos % 5 == 4:
                    filename = os.path.join(folder_val, f'wideo_{building_name}_val_{saved_val}.jpg')
                    saved_val += 1
                else:
                    filename = os.path.join(folder_dataset, f'wideo_{building_name}_val_{saved_dataset}.jpg')
                    saved_dataset += 1

                #zapisanie pliku
                cv2.imwrite(filename, frame)

            fps_id += 1

        cap.release()

print(f"Gotowe dla {building_name}")
print(f"Zapisano {saved_dataset} zdjec do {folder_dataset}")
print(f"Zapisano {saved_val} zdjec do {folder_val}")

