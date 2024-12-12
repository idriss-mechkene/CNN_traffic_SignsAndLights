import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

yolo_model = YOLO('yolo_model.onnx')  

classification_model = load_model('traffic_signs_model_90x90_crop_94_17.keras')


CLASS_LABELS = [
    'Class 0: Green Light',
    'Class 1: Red Light',
    'Class 2: Speed Limit 10',
    'Class 3: Speed Limit 100',
    'Class 4: Speed Limit 110',
    'Class 5: Speed Limit 120',
    'Class 6: Speed Limit 20',
    'Class 7: Speed Limit 30',
    'Class 8: Speed Limit 40',
    'Class 9: Speed Limit 50',
    'Class 10: Speed Limit 60',
    'Class 11: Speed Limit 70',
    'Class 12: Speed Limit 80',
    'Class 13: Speed Limit 90',
    'Class 14: Stop'
]

def crop_and_classify(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = yolo_model(image_rgb)

    cropped_results = []
    for detection in results[0].boxes:
        x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])

        cropped_img = image_rgb[y_min:y_max, x_min:x_max]

       # cropped_img_resized = cv2.resize(cropped_img, (416, 416))
        cropped_img_resized = cv2.resize(cropped_img, (90, 90))
        img_array = img_to_array(cropped_img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = classification_model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        class_label = CLASS_LABELS[class_idx]
        confidence = float(predictions[0][class_idx])

        cropped_results.append({
            "label": class_label,
            "confidence": confidence,
            "bbox": (x_min, y_min, x_max, y_max),
            "image": cropped_img
        })

    return cropped_results

def main():
    st.title("Traffic Sign & Light Detector")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.read())

        results = crop_and_classify("temp_image.jpg")

        for result in results:
            st.image(result["image"], caption=f"{result['label']} ({result['confidence']:.2f})")
            st.write(f"Bounding Box: {result['bbox']}")
            st.write("---")

if __name__ == "__main__":
    main()
