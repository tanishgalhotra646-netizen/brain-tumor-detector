import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

# ==========================================
# TITLE
# ==========================================

st.title("🧠 Brain Tumor Classification")
st.write(
    "Upload an MRI image to predict the brain tumor category."
)

st.warning(
    "⚠️ This application is for educational and research purposes only. "
    "It is not a medical diagnosis."
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "brain_tumor_model.keras"
    )
    return model


model = load_model()

# ==========================================
# CLASS NAMES
# ==========================================

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📤 Upload an MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    # Open image and force RGB format
    image = Image.open(uploaded_file).convert("RGB")

    # Display original image
    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    # Resize image
    image_resized = image.resize((224, 224))

    # Convert image to NumPy array
    img_array = np.array(image_resized)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Convert to float32
    img_array = img_array.astype("float32")

    # ==========================================
    # CHECK INPUT SHAPE
    # ==========================================

    st.write(
        "Input Shape:",
        img_array.shape
    )

    # ==========================================
    # MODEL PREDICTION
    # ==========================================

    prediction = model.predict(
        img_array,
        verbose=0
    )

    # Get predicted class index
    predicted_class = np.argmax(prediction[0])

    # Get confidence
    confidence = np.max(prediction[0])

    # Get predicted class name
    predicted_label = class_names[predicted_class]

    # ==========================================
    # DISPLAY RESULT
    # ==========================================

    st.success(
        f"🧠 Prediction: {predicted_label}"
    )

    st.info(
        f"📊 Confidence: {confidence * 100:.2f}%"
    )

    # ==========================================
    # DISPLAY ALL CLASS PROBABILITIES
    # ==========================================

    st.subheader("Prediction Probabilities")

    for i, class_name in enumerate(class_names):

        probability = prediction[0][i] * 100

        st.write(
            f"{class_name}: {probability:.2f}%"
        )

        st.progress(
            float(prediction[0][i])
        )
