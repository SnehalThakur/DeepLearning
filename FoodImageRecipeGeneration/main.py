import streamlit as st

import streamlit as st
import tensorflow as tf
import keras.utils as image
import numpy as np
from PIL import Image, ImageOps  # Streamlit works with PIL library very easily for Images
import cv2
import utils.SQLiteDB as dbHandler
from app import prediction
import os

model_path = '.\\pretrainedWeights\\custom_model.h5py'

def save_uploadedfile(uploadedfile, path):
    with open(path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    print("Saved File:{} to upload".format(uploadedfile.name))


st.title("Food Image Recipe Generation")
upload = st.file_uploader('Upload a food image')

if upload is not None:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)

    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # Color from BGR to RGB
    print("type of opencv",type(opencv_image))
    img = Image.open(upload)
    st.image(img, caption='Uploaded Image', width=300)
    if st.button('Predict'):
        # Load pretrained Model
        model = tf.keras.models.load_model(model_path)

        path_dir = os.path.join(os.getcwd(), 'upload')
        print("path_dir =",path_dir)
        upload_path = os.path.join(path_dir, upload.name)
        print("upload_path=",upload_path)
        imagePath='.//dataset//food11//test//cheesecake//305424.jpg'

        # Save uploaded file
        save_uploadedfile(upload, upload_path)

        # Prediction on uploaded image
        result = prediction(model, upload_path)
        st.title(result)

        # retrieve recipe from DB
        recipe = dbHandler.retrieveRecipeDataWithItemName(result)
        st.text_area(label ="Recipe",value=recipe[0][2], height =500)
