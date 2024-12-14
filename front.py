import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import os
import numpy as np

# Define the directory containing the labels
LABELS_DIR = r'C:\Users\idris\OneDrive\Documents\DataSet\test\labels'

# Load the model
model = load_model('traffic_signs_model_90x90_crop_94_17.keras')

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

def predict_and_crop_image(image, label_path):
    img = Image.open(image)
    img_width, img_height = img.size

    cropped_images = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id, x_center, y_center, width, height = map(float, line.split())

                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height

                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)

                cropped_img = img.crop((x_min, y_min, x_max, y_max))
                #cropped_img = cropped_img.resize((416, 416))
                cropped_img = cropped_img.resize((90, 90))
                cropped_img_array = img_to_array(cropped_img) / 255.0
                st.image(cropped_img_array, caption='Uploaded Image.')

                cropped_images.append(cropped_img_array)

    cropped_images = np.array(cropped_images)

    predictions = model.predict(cropped_images)
    results = []
    for pred in predictions:
        class_idx = np.argmax(pred)
        class_label = CLASS_LABELS[class_idx]
        confidence = float(pred[class_idx])
        results.append((class_label, confidence))

    return results

def main():
    st.title("Traffic Sign Classifier with corp")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.')

        file_name = os.path.basename(uploaded_file.name)
        label_name = file_name.replace('.jpg', '.txt')
        label_path = os.path.join(LABELS_DIR, label_name)

        if os.path.exists(label_path):
            results = predict_and_crop_image(uploaded_file, label_path)

            for idx, (class_label, confidence) in enumerate(results):
                st.write(f"Crop {idx + 1}:")
                st.write(f"Predicted Class: {class_label}")
                st.write(f"Confidence: {confidence:.2f}")
        else:
            st.error(f"Label file not found for the image: {label_name}")

if __name__ == "__main__":
    main()
