# detection islemi icin kullanacağımız metod burada

# library
import cv2
import numpy as np  # goruntulerdeki tip donusumu
from ultralytics import YOLO

# detection function
def detect_plate(image, model_path):
    # OpenCV font tanımlaması
    font = cv2.FONT_HERSHEY_SIMPLEX

    print("[INFO].. Image is loading!")
    image_array = np.asarray(image).copy()  # Görüntüyü numpy array'e çevirip kopyaladık
    image_array.setflags(write=1)  # Numpy dizisinin salt okunur olmasını kaldır
    print("[INFO].. Processing is started!")
    
    # Modeli yükle
    model = YOLO(model_path)
    results = model(image_array)[0]

    # Kontrol: len yani uzunluk 0'sa boş demektir
    is_detected = len(results.boxes.data.tolist())
    
    if is_detected != 0:
        threshold = 0.5  # eşik
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
      
            if score > threshold:
                cropped_image = image_array[y1:y2, x1:x2]

                # Tespit edilen alanı çerçevele
                cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Sınıf ismini ve skorunu ekle
                class_name = results.names[class_id]
                score = score * 100
                text1 = f"{class_name}: %{score:.2f}"
                cv2.putText(image_array, text1, (x1, y1 - 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    else:  # Eğer detection işlemi yoksa, görsel üzerine 'no detection' yaz
        text = "no detection"
        cv2.putText(image_array, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return image_array, cropped_image if is_detected != 0 else None, is_detected
