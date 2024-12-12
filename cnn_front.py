import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the model
model = load_model('traffic_signs_model_without_crop_64_29.keras')

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

def predict_image(image):
    # Process the image
    image = load_img(image, target_size=(416, 416))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(image)
    print(predictions)
    class_idx = np.argmax(predictions[0])
    class_label = CLASS_LABELS[class_idx]

    return class_label, float(predictions[0][class_idx])

def main():
    st.title("Traffic Sign & light")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Predict the class
        class_label, confidence = predict_image(uploaded_file)

        # Display the result
        st.write(f"Predicted Class: {class_label}")
        st.write(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
