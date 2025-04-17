import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


model = load_model("glaucoma_classifier_m_EfficientNetB0.h5")

st.title("🧠 Glaucoma Detection App")
st.write("Upload a retinal image to check for signs of glaucoma.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)


    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]
    if prediction >= 0.5:
        st.error("⚠️ Glaucoma Detected")
    else:
        st.success("✅ No Glaucoma Detected")
