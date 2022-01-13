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

def asciiart(in_f, SC, GCF,  out_f, color1='black', color2='blue', bgcolor='white'):

    # The array of ascii symbols from white to black
    chars = np.asarray(list(' .,:irs?@9B&#'))

    # Load the fonts and then get the the height and width of a typical symbol 
    # You can use different fonts here
    font = ImageFont.load_default()
    letter_width = font.getsize("x")[0]
    letter_height = font.getsize("x")[1]

    WCF = letter_height/letter_width

    #open the input file
    img = Image.open(in_f)


    #Based on the desired output image size, calculate how many ascii letters are needed on the width and height
    widthByLetter=round(img.size[0]*SC*WCF)
    heightByLetter = round(img.size[1]*SC)
    S = (widthByLetter, heightByLetter)

    #Resize the image based on the symbol width and height
    img = img.resize(S)
    
    #Get the RGB color values of each sampled pixel point and convert them to graycolor using the average method.
    # Refer to https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/ to know about the algorithm
    img = np.sum(np.asarray(img), axis=2)
    
    # Normalize the results, enhance and reduce the brightness contrast. 
    # Map grayscale values to bins of symbols
    img -= img.min()
    img = (1.0 - img/img.max())**GCF*(chars.size-1)
    
    # Generate the ascii art symbols 
    lines = ("\n".join( ("".join(r) for r in chars[img.astype(int)]) )).split("\n")

    # Create gradient color bins
    nbins = len(lines)
    #colorRange =list(Color(color1).range_to(Color(color2), nbins))

    #Create an image object, set its width and height
    newImg_width= letter_width *widthByLetter
    newImg_height = letter_height * heightByLetter
    newImg = Image.new("RGBA", (newImg_width, newImg_height), bgcolor)
    draw = ImageDraw.Draw(newImg)

    # Print symbols to image
    leftpadding=0
    y = 0
    lineIdx=0
    for line in lines:
        color = 'blue'
        lineIdx +=1

        draw.text((leftpadding, y), line, '#0000FF', font=font)
        y += letter_height

    # Save the image file

    #out_f = out_f.resize((1280,720))
    newImg.save(out_f)


def load_image(filename, size=(512,512)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels


def imgGen2(img1):
  inputf = img1  # Input image file name

  SC = 0.1    # pixel sampling rate in width
  GCF= 2      # contrast adjustment

  asciiart(inputf, SC, GCF, "results.png")   #default color, black to blue
  asciiart(inputf, SC, GCF, "results_pink.png","blue","pink")
  img = Image.open(img1)
  img2 = Image.open('results.png').resize(img.size)
  #img2.save('result.png')
  #img3 = Image.open('results_pink.png').resize(img.size)
  #img3.save('resultp.png')
  return img2




	
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
        if Image is not None:
		
	#src_image = load_image(Image)
		image = Image.open(Image)	

		st.image(Image, caption='Input Image', use_column_width=True)
			#st.write(os.listdir())
		im = imgGen2(Image)	
		st.image(im, caption='ASCII art', use_column_width=True) 

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
