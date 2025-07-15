import streamlit as st
import cv2
import numpy as np
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model and train generator (for class labels)
model = joblib.load('potatoe_classifier.pkl')

# Load train generator just to get class labels (assuming same structure as before)
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'data/Train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Streamlit app
st.title("Potato Blight Checker")
st.write("Upload an image of a potato leaf to check its health status")

def predict_image(img):
    img = cv2.resize(img, (150, 150))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    class_label = list(train_generator.class_indices.keys())[class_idx]
    confidence = pred[0][class_idx]
    return class_label, confidence

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display the image
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction
    label, confidence = predict_image(img)
    
    # Show results
    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2%}")
    
    # Optional: show probability distribution
    st.subheader("Class Probabilities")
    classes = list(train_generator.class_indices.keys())
    probs = model.predict(np.expand_dims(cv2.resize(img, (150, 150))/255.0, axis=0))[0]
    for cls, prob in zip(classes, probs):
        st.write(f"{cls}: {prob:.2%}")
