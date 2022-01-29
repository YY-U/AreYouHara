import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras_facenet import FaceNet


st.title('Similarity with Hara')
st.write("早速あなたの顔の画像をアップロードしてみましょう")

uploaded_file = st.file_uploader('Choose a image file')

facenet = FaceNet()

embeddings_hara = np.load('embeddings_hara.npy') # 顔ベクトルの読み込み

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image = ImageOps.exif_transpose(image)  # 画像を適切な向きに補正する
    img_array = np.array(image)

    extracts = facenet.extract(img_array)

    # 顔が画像中に存在しないとき
    if len(extracts) < 1:
        st.subheader('Face detection faild')
    else:
        max_extract = max(extracts, key=lambda x:x['box'][2]*x['box'][3]) # もっとも大きい顔を取得

        embed_img = max_extract['embedding']

        distance = facenet.compute_distance(embeddings_hara, embed_img)
        st.subheader(distance)

        # 切り取った顔画像を取得
        x,y,width,hight = max_extract['box']
        img_array = img_array[y:y+hight, x:x+width]

    st.image(
        img_array, caption='upload images',
        use_column_width=True
    )
