import os
from ulti import *
import threading
from PIL import Image
import requests
from io import BytesIO
import streamlit as st
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
# from streamlit_webrtc import (
#     AudioProcessorBase,
#     ClientSettings,
#     VideoProcessorBase,
#     WebRtcMode,
#     webrtc_streamer,
# )
# import av

# WEBRTC_CLIENT_SETTINGS = ClientSettings(
#     rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#     media_stream_constraints={
#         "video": True,
#         "audio": False,
#     },)

model_path = os.path.join('model','ModelTrainOnKaggle.h5')
@st.cache
def model_load():
    model = tf.keras.models.load_model(model_path)
    return model





	
def main():
    st.set_page_config(layout="wide")
    
    st.image(os.path.join('Images','Banner No2.png'), use_column_width  = True)
    st.markdown("<h1 style='text-align: center; color: white;'>是時候改變不同風格了~</h1>", unsafe_allow_html=True)
    with st.beta_expander("Configuration Option"):

        st.write("**AutoCrop** help the model by finding and cropping the biggest face it can find.")
        st.write("**Gamma Adjustment** can be used to lighten/darken the image")
    comic_model = model_load()

    menu = ['Image Based', 'URL']
    #menu = ['本機照片']
    st.sidebar.header('照片上傳選擇')
    choice = st.sidebar.selectbox('選擇上傳方式 ?', menu)

    if choice == 'Image Based':
        st.sidebar.header('配置')
        outputsize = st.sidebar.selectbox('輸出尺寸', [384,512,768])
        Autocrop = st.sidebar.checkbox('自動裁剪照片',value=True) 
        gamma = st.sidebar.slider('Gamma 調整', min_value=0.1, max_value=3.0,value=1.0,step=0.1) # change the value here to get different result
        
        Image = st.file_uploader('在這上傳您的檔案',type=['jpg','jpeg','png'])

        if file_uploader is not None:
            col1, col2 = st.beta_columns(2)
            Image = Image.read()
            Image = tf.image.decode_image(Image, channels=3).numpy()                  
            Image = adjust_gamma(Image, gamma=gamma)
            with col1:
                st.image(Image)
            input_image = loadtest(Image,cropornot=Autocrop)
            prediction = comic_model(input_image, training=True)
            prediction = tf.squeeze(prediction,0)
            prediction = prediction* 0.5 + 0.5
            prediction = tf.image.resize(prediction, 
                [outputsize, outputsize],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            prediction=  prediction.numpy()
            with col2:
                st.image(prediction)

    elif choice == 'URL':
        st.sidebar.header('配置')

        outputsize = st.sidebar.selectbox('輸出尺寸', [384,512,768])
        Autocrop = st.sidebar.checkbox('自動裁剪照片',value=True) 
        gamma = st.sidebar.slider('Gamma 調整', min_value=0.1, max_value=3.0,value=1.0,step=0.1) # change the value here to get different result
        
         
        url = st.text_input('網址連結')
        response = requests.get(url)
        Image = (response.content)
        if Image is not None:
            col1, col2 = st.beta_columns(2)
            Image = tf.image.decode_image(Image).numpy()
            Image = adjust_gamma(Image, gamma=gamma)
            with col1:
                st.image(Image)
            text_input = loadtest(Image,cropornot=Autocrop)
            prediction = comic_model(text_input, training=True)
            prediction = tf.squeeze(prediction,0)
            prediction = prediction* 0.5 + 0.5
            prediction = tf.image.resize(prediction, 
                         [outputsize, outputsize],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            prediction=  prediction.numpy()
            with col2:
                st.image(prediction)
 

if __name__ == '__main__':
    main()
