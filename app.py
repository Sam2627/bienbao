import streamlit as st
from PIL import Image, ImageOps
from img_classification import teachable_machine_classification

st.title("Image Classification with Google's Teachable Machine")
st.header("Phát hiện biển báo giao thông")
st.text("Upload a scan for Classification")

uploaded_file = st.file_uploader("Choose a scan ...", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Scan.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'model/keras_model.h5')
    if label == 0:
        st.write("Biển dừng")
    elif label == 1:
        st.write("Biển cấm xe tải")
    elif label == 2:
        st.write("Biển cấm ô tô")
    elif label == 3:
        st.write("Biển chỗ quay xe")
    elif label == 4:
        st.write("Biển cấm đi ngược chiều")
    else:
        st.write("Phát hiện lỗi!")
