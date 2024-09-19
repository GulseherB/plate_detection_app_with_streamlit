# library
import cv2
from PIL import Image
import streamlit as st
from helper import detect_plate

# title
st.title("Plate Recognition System 🚗")

# header (alt başlık)
st.header("Upload an image")

# files (dosyayı web sitesine yükleme çubuğunu oluştur)
file = st.file_uploader("", type=["png", "jpg", "jpeg"])

# model path
model_path = r"C:\Users\seherb\Desktop\number_plate_reading_app\models\plate_detection_model.pt"

# image (resmi okuma işlemi)
if file is not None:
    # original image
    st.header("Original image:")
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # processed image
    st.header("Detection result:")
    # function call
    detection_result, cropped_image, is_detected = detect_plate(image, model_path)

    if is_detected != 0:
        st.write("#### [INFO].. Plate is detected:")
        st.image(detection_result, use_column_width=True)
        if cropped_image is not None:
            st.image(cropped_image, use_column_width=True)
    else:
        st.write("#### [INFO].. Plate is not detected!")
        st.image(detection_result, use_column_width=True)
