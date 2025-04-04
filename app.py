import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model



@st.cache_resource
def load_model_cached():
    return load_model('best_model.keras')

model = load_model_cached()

categories = ['Auto Rickshaws', 'Bikes', 'Car', 'Motorcycles', 'Planes', 'Ships', 'Trains']

st.title('Image Classification Web App')

st.markdown("### Kéo thả ảnh hoặc nhấn vào nút để tải ảnh lên:")
uploaded_file = st.file_uploader("chọn hình ảnh...", type=["jpg", "png", "jpeg"])
st.markdown("""
<style>
    .rounded {
        border-radius: 15px;
    } 
</style>
""", unsafe_allow_html=True)
if uploaded_file is not None:
   
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    
    img_height, img_width = 150, 150  
    image = image.resize((img_width, img_height))
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  

    
    predictions = model.predict(image)

    predicted_class_index = np.argmax(predictions[0])
    predicted_class = categories[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    if confidence < 60:
        st.write("Không phải phương tiện.")
    else:
        st.write(f"Dự đoán: **{predicted_class}** có **{confidence:.2f}%** đúng.")

    st.write("Kết quả dự đoán:")
    st.markdown("<h4>Kết quả dự đoán:</h4>", unsafe_allow_html=True)
    
   
    st.markdown("<h4>Kết quả dự đoán:</h4>", unsafe_allow_html=True)
    result_df = {cat: f"{predictions[0][i] * 100:.2f}%" for i, cat in enumerate(categories)}
    st.table(result_df)

   
    fig, ax = plt.subplots()
    ax.barh(categories, predictions[0]) 
    ax.set_xlim(0, 1)
    ax.set_xlabel('Xác suất dự đoán')
    st.pyplot(fig)

url = st.text_input("Enter image URL...")
st.text('Cuối liên kết URL phải có ."jpg", "png", "jpeg"')

if url:
    if not (url.startswith('http://') or url.startswith('https://')):
        st.error("Please enter a valid URL starting with 'http://' or 'https://' ,cuối liên kết URL phải có .jpg, png, jpeg.")
    else:
        try:
            # Tải ảnh từ URL
            response = requests.get(url)
            response.raise_for_status()  
            image = Image.open(BytesIO(response.content)).convert('RGB')
            st.image(image, caption='Image from URL', use_column_width=True)

            # Tiền xử lý ảnh để dự đoán
            img_height, img_width = 150, 150  
            image = image.resize((img_width, img_height))
            img_array = np.array(image) / 255.0 
            
           
            img_array = np.expand_dims(img_array, axis=0)

            # Dự đoán nhãn của ảnh
            predictions = model.predict(img_array)

            predicted_class_index = np.argmax(predictions[0])
            predicted_class = categories[predicted_class_index]
            confidence = predictions[0][predicted_class_index] * 100

            # Hiển thị kết quả dự đoán
            st.write(f"Dự đoán: **{predicted_class}** có **{confidence:.2f}%** đúng.")
            st.markdown("<h4>Kết quả dự đoán:</h4>", unsafe_allow_html=True)
            result_df = {cat: f"{predictions[0][i] * 100:.2f}%" for i, cat in enumerate(categories)}
            st.table(result_df)

            # Hiển thị biểu đồ thanh
            fig, ax = plt.subplots()
            ax.barh(categories, predictions[0]) 
            ax.set_xlim(0, 1)
            ax.set_xlabel('Xác suất dự đoán')
            st.pyplot(fig)
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.write("Vui lòng nhập URL để nhận dự đoán")
