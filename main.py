# from mira.detectors import FasterRCNN
from mtcnn import MTCNN
from keras_facenet import FaceNet
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

face_dir = "hara"

embeddings = [] # 顔ベクトル（顔特徴量）
imgs = [] # 顔領域を切り取ったもの
detector = MTCNN() # 顔領域の検出器
embedder = FaceNet() # FaceNetモデル

def extract(face, img):
    x,y,width,hight = face['box']
    return img[y:y+hight, x:x+width]

files = os.listdir(face_dir) # ディレクトリ のファイルリストを取得
for file in files:
    file_path = os.path.join(face_dir, file)
    print(file_path)
    img = cv2.imread(file_path) # 画像読み込み
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB形式に変換
    faces = detector.detect_faces(img_rgb) # 顔領域を検出．画像中に複数の顔が検出されることも想定する
    max_face = max(faces, key=lambda face:face['box'][2]*face['box'][3])
    embedding = embedder.embeddings([extract(max_face, img)]) # 潜在変数表現に変換

    embeddings.append(embedding[0])
    imgs.append(extract(max_face, img)) # 顔領域を保存しておく

for i,img in enumerate(imgs):
    cv2.imwrite(f'extracted_hara/{i}.jpg', img)

np.save('embeddings_hara', np.array(embeddings).mean(axis=0))
