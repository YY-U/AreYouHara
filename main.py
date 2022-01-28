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
file_paths = []
facenet = FaceNet() # FaceNetモデル

def extract(face, img):
    x,y,width,hight = face['box']
    return img[y:y+hight, x:x+width]

files = os.listdir(face_dir) # ディレクトリ のファイルリストを取得
for file in files:
    file_path = os.path.join(face_dir, file)
    print(file_path)

    extracts = facenet.extract(file_path)
    if len(extracts) < 1:
        continue
    max_extract = max(extracts, key=lambda x:x['box'][2]*x['box'][3])
    embeddings.append(max_extract['embedding']) 

    file_paths.append(file_path)

np.save('file_paths', np.array(file_paths))

np.save('embeddings_hara', np.array(embeddings).mean(axis=0))
