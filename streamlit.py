import streamlit as st
from PIL import Image
import io 
import numpy as np
from keras_facenet import FaceNet


facenet = FaceNet()
embeddings_hara = np.load('embeddings_hara.npy')

st.title('Hello')

uploaded_file = st.file_uploader('Choose a image file')

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)

    extracts = facenet.extract(img_array)
    max_extract = max(extracts, key=lambda x:x['box'][2]*x['box'][3])

    embed_img = max_extract['embedding']

    distance = facenet.compute_distance(embeddings_hara, embed_img)
    st.subheader(distance)

    st.image(
        image, caption='upload images',
        use_column_width=True
    )