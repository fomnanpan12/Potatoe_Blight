import streamlit as st
import cv2
import numpy as np
import joblib

# Load model and classes
saved_data = joblib.load('potato_classifier.pkl')
model = saved_data['model']
class_names = saved_data['class_names']

st.title("Potato Blight Checker")
st.write("Upload an image of a potato leaf to check its health status")

def predict_image(img):
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    class_idx = np.argmax(pred)
    return class_names[class_idx], pred[class_idx]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
    label, confidence = predict_image(img)
    
    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2%}")
    
    st.subheader("Class Probabilities")
    probs = model.predict(np.expand_dims(cv2.resize(img, (150, 150))/255.0, axis=0))[0]
    for cls, prob in zip(class_names, probs):
        st.write(f"{cls}: {prob:.2%}")