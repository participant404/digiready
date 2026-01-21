#dependencies
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import time



#functions
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for Teachable Machine TFLite model.
    - Resize to 224x224
    - Convert to float32
    - Normalize to [-1, 1]
    - Add batch dimension
    """
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    image_array = np.asarray(image).astype(np.float32)
    image_array = (image_array / 127.5) - 1
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

def predict_image(image_array: np.ndarray, interpreter, labels: list) -> dict:
    """
    Run TFLite inference and return prediction results.
    """
    if interpreter is not None:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    else:
        input_details = None
        output_details = None
    interpreter.set_tensor(input_details[0]["index"], image_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    class_index = int(np.argmax(output))
    confidence = float(output[0][class_index])
    return {
        "class": labels[class_index],
        "confidence": confidence,
        "all_confidences": output[0]
    }

def load_labels_from_upload(uploaded_file) -> list:
    uploaded_file.seek(0)
    try:
        content = uploaded_file.read().decode("utf-8").splitlines()
        labels = [line.strip().split(" ", 1)[1] for line in content if line.strip()]
        if len(labels) == 0:
            raise ValueError("Labels file is empty.")
        return labels
    except Exception as e:
        st.error(f"Failed to read labels file: {e}")
        return []

@st.cache_data
def load_labels(path: str) -> list:
    with open(path, "r") as f:
        return [line.strip().split(" ", 1)[1] for line in f]

@st.cache_resource
def load_interpreter(model_path: str):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    return interpreter



#labels and interpreter
interpreter = None
labels = None
if st.session_state.get("model_source", "sample") == "sample":
    labels = load_labels("labels.txt")
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
elif st.session_state.get("model_source") == "uploaded":
    if "uploaded_model" in st.session_state and "uploaded_labels" in st.session_state:
        st.session_state.uploaded_model.seek(0)
        st.session_state.uploaded_labels.seek(0)
        labels = load_labels_from_upload(st.session_state.uploaded_labels)
        try:
            interpreter = tf.lite.Interpreter(
                model_content=st.session_state.uploaded_model.read()
            )
            interpreter.allocate_tensors()
        except Exception as e:
            st.error(f"Failed to load model: {e}\nMake sure it’s a Teachable Machine TensorFlow Lite model.")
            interpreter = None
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



#initialise persistent variables
if "page" not in st.session_state:
    st.session_state.page = "model_select"
if "upload_time" not in st.session_state:
    st.session_state.upload_time = 0
    st.session_state.upload_image = None
if "camera_time" not in st.session_state:
    st.session_state.camera_time = 0
    st.session_state.camera_image = None
if "last_image" not in st.session_state:
    st.session_state.last_image = None
if "model_source" not in st.session_state:
    st.session_state.model_source = "sample"  # or "uploaded"



#website



#model selector
if st.session_state.page == "model_select":
    #title
    st.markdown(
        """
        <div style="
            background-color:#ACB0EF; 
            padding:5px;
            border-radius:5px;
            text-align:center;
            font-size:40px;
        ">
            Teachable Machine Image Classifier
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""<br>""", unsafe_allow_html=True)



    #instructions
    with st.expander("""### How do I create a model from Teachable Machine?"""):
        st.markdown("""
            - Go to [Teachable Machine](https://teachablemachine.withgoogle.com/)
            - Choose **Image Project** with **Standard Image Model**
            - Train model using webcam or uploaded photos
            - Export as:
                - **TensorFlow Lite**
                - **Model conversion type: Floating point**
            - Download and extract the `model_uquant.tflite` and `labels.txt` files
            - Upload them below:
            """
        )
    st.markdown("""<br>""", unsafe_allow_html=True)


    #uploads
    model_file = st.file_uploader(
        "Upload a TensorFlow Lite model (.tflite)",
        type=["tflite"]
    )
    labels_file = st.file_uploader(
        "Upload labels file (labels.txt)",
        type=["txt"]
    )
    st.divider()

    #action buttons
    col1, col2 = st.columns(2)
    with col1:
        use_model = st.button("Use uploaded model")
    with col2:
        use_sample = st.button("Use sample model")
if st.session_state.page == "model_select":
    if use_sample:
        # st.session_state.model_path = "model.tflite"
        # st.session_state.labels_path = "labels.txt"
        st.session_state.model_source = "sample"
        st.session_state.page = "predict"
        st.rerun()
    if use_model:
        if model_file is None or labels_file is None:
            st.error("Please upload both a .tflite model and a labels.txt file.")
        else:
            st.session_state.model_source = "uploaded"
            st.session_state.uploaded_model = model_file
            st.session_state.uploaded_labels = labels_file
            st.session_state.page = "predict"
            st.rerun()



#predictor
if st.session_state.page == "predict":
    #title
    st.title("Image Classifier")
    st.write("Upload an image or take a photo to classify it.")



    #back button
    if st.button("← Back to model selection"):
        st.session_state.page = "model_select"
        st.session_state.last_image = None
        st.session_state.last_output = None
        st.session_state.upload_image = None
        st.session_state.camera_image = None
        st.session_state.upload_time = 0
        st.session_state.camera_time = 0
        if "uploaded_model" in st.session_state:
            del st.session_state.uploaded_model
        if "uploaded_labels" in st.session_state:
            del st.session_state.uploaded_labels
        st.rerun()



    #get latest image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    camera_image = st.camera_input("Or take a photo")
    if uploaded_file is not None and uploaded_file != st.session_state.upload_image:
        st.session_state.upload_time = time.time()
        st.session_state.upload_image = uploaded_file
    if camera_image is not None and camera_image != st.session_state.camera_image:
        st.session_state.camera_time = time.time()
        st.session_state.camera_image = camera_image
    latest_image = None
    upload_time = st.session_state.get("upload_time", 0)
    camera_time = st.session_state.get("camera_time", 0)
    if upload_time > camera_time:
        latest_image = st.session_state.upload_image
    elif camera_time > upload_time:
        latest_image = st.session_state.camera_image
    if latest_image is not None:
        st.session_state.last_image = latest_image



    #predict using image then display
    if st.session_state.last_image is not None:
        output_container = st.container()
        try:
            image = Image.open(st.session_state.last_image).convert("RGB")

            image_array = preprocess_image(image)
            result = predict_image(image_array, interpreter, labels)

            st.session_state.last_output = result
            with output_container:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.subheader("Latest Prediction")
                st.image(image, width=1000)
                st.markdown(f"**Prediction:** {st.session_state.last_output['class']}")
                st.markdown(f"**Confidence:** {st.session_state.last_output['confidence']:.2%}")
                with st.expander("See all class confidence levels"):
                    for i, score in enumerate(st.session_state.last_output["all_confidences"]):
                        col1, col2, col3 = st.columns([1, 1, 4])
                        with col1:
                            st.write(f"{labels[i]}")
                        with col2:
                            st.write(f"{score:.2%}")
                        with col3:
                            st.progress(int(score*100))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    #spacing
    st.markdown("<br><br><br>", unsafe_allow_html=True)



#style.css but in python
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #ACB0EF;  /* blue */
        color: black;               /* text color */
        height: 3.5em;                /* taller button */
        width: 100%;                /* fill column width */
        font-size:18px;
        border-radius: 10px;        /* rounded corners */
    }
    .stApp {
        background-color: #f0f4f8;  /* soft light blue-gray */
    }
    </style>
    """,
    unsafe_allow_html=True
)
