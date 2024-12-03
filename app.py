from file import ValueSet
import json, time
from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
# import numpy as np
import cv2
# importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
#pip install facenet-pytorch


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow import keras
from file import ValueSet
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from flask import *
import json, time
INFANT_HITTING=0
INFANT_CRYING=0
INFANT_CHOKING=0

obj= ValueSet()


app=Flask(__name__)
# @app.route('/',methods=['GET'])

# def home_page():
#     val1=obj.get_value3()
#     val2=obj.get_value4()
#     val3=obj.get_value5()

#     data_set= {
#         'result_infant_choking': val1,
#         'result_infant_crying': val2,
#         'result_infant_hitting': val3,
#         'Timestamp':time.time(),
#         }
#     json_dump=json.dumps(data_set)
#     return json_dump

@app.route('/data',methods=['GET'])
#@cross_origin(origins=['http://127.0.0.1:5000'])
def home_page():
    
    val4=obj.get_value3()
    val5=obj.get_value4()
    val6=obj.get_value5()
    
    val1 = obj.get_value1()
    val2 = obj.get_value2()
    
    data_set= {
        'result_infant_hitting': val6,
        'result_infant_chocking': val4,
        'result_infant_crying': val5,
        'result_infant_facerecognition': val1,
        'result_infant_sharp objects': val2,
        'Timestamp':time.time(),
        }
    json_dump=json.dumps(data_set)
    return json_dump


list=[]

sequence_model = pickle.load(open('finalized_model.sav', 'rb'))

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

IMG_SIZE = 224


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()

label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_df["tag"]))
print(label_processor.get_vocabulary())

labels = train_df["tag"].values
labels = label_processor(labels[..., None]).numpy()
labels

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 5

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("test", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]
    j=0
    for i in np.argsort(probabilities)[::-1]:
        #list[j].append(probabilities[i]) 
        list.append(f"  {class_vocab[i]}:{probabilities[i] * 100:5.2f}")
        #list.append(1)
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    
    
    name=[]
    for i in range(0,3):
        name.append(list[i].split(":"))
        
        if(name[i][0]=="  INFANT_CRYING"):
            INFANT_CRYING=float(name[i][1])
            
            print(INFANT_CRYING)
        elif(name[i][0]=="  INFANT_CHOKING"):
            INFANT_CHOKING=float(name[i][1])
            
            
        elif(name[i][0]=="  INFANT_HITTING"):
            INFANT_HITTING=float(name[i][1])
            
    x1=0
    x2=0
    x3=0

    if(INFANT_CHOKING>93):
        x1=1
        obj.set_value3(x1)
    if(INFANT_CRYING>93):
        x2=1
        obj.set_value4(x2)
    if(INFANT_HITTING>93):
        x3=1
        obj.set_value5(x3)
    else:
        x1=x2=x3=0
    
    return frames
        


test_video = "/Users/amitsai/Desktop/cnn_lstm/show_videos/normal/pexels-william-fortunato-6391719.mp"
"/Users/amitsai/Desktop/cnn_lstm/show_videos/Crying/96.mp4"
"/Users/amitsai/Desktop/cnn_lstm/show_videos/choking/34.mp4"
"""/Users/amitsai/Desktop/cnn_lstm/show_videos/Crying/96.mp4"""
"/Users/amitsai/Desktop/cnn_lstm/show_videos/normal/pexels-william-fortunato-6391719.mp4"
"""
/Users/amitsai/Desktop/cnn_lstm/show_videos/hitting/VID_20220910_223124.mp4"""
print(f"Test video path: {test_video}")

test_frames = sequence_prediction(test_video)



mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion
def face_match(img_path, data_path): # img_path= location of photo, data_path= location of data.pt 
    # getting embedding matrix of the given img
    img = Image.open(img_path)
    face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
    emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
    
    saved_data = torch.load('data.pt') # loading data.pt file
    embedding_list = saved_data[0] # getting embedding data
    name_list = saved_data[1] # getting list of names
    dist_list = [] # list of matched distances, minimum distance is used to identify the person
    
    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)
        
    idx_min = dist_list.index(min(dist_list))
    if(min(dist_list)>0.8): 
        name = "Unknown"
        return(name)
    else:
        return (name_list[idx_min], min(dist_list))

# obj = ValueSet()
# app=Flask(__name__)


camera = cv2.VideoCapture(0)

process_this_frame = True
x=0
infant_hitting=0
infant_crying=0
infant_chocking=0
img_number=1


# @app.route('/data',methods=['GET'])
# #@cross_origin(origins=['http://127.0.0.1:5000'])
# def home_page():
#     global infant_hitting
#     global infant_crying
#     global infant_chocking
#     val1 = obj.get_value1()
#     val2 = obj.get_value2()
    
    
#     print(val1)
#     print(val2)
    
#     data_set= {
#         'result_infant_hitting': infant_hitting,
#         'result_infant_chocking': infant_chocking,
#         'result_infant_crying': infant_crying,
#         'result_infant_facerecognition': val1,
#         'result_infant_sharp objects': val2,
#         'Timestamp':time.time(),
#         }
#     json_dump=json.dumps(data_set)
#     return json_dump

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames() 


def gen_frames():  
    while True:
    # Extract a frame
        _, frame = camera.read()
        frame = cv2.resize(frame, None, fx=0.9, fy=0.9)
        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        person="person"
        scissors = "Scissors"
        knife = "Knife"
        fork="fork"
        global x
        x=0

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    if(class_id+1 == 1): 
                        print(person)
                        cv2.putText(frame, "person:detected", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        img1 = "./data/"+str(img_number)+".jpg"
                        cv2.imwrite(img1, frame)
                        
                        face_classifier = cv2.CascadeClassifier('/Users/amitsai/Desktop/weights/Haarcascades/haarcascade_frontalface_default.xml')

                            # Load our image then convert it to grayscale
                        image = cv2.imread('/Users/amitsai/Desktop/ngrok/finalcode/data/1.jpg')
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                            
                        faces = face_classifier.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)

                            # When no faces detected, face_classifier returns and empty tuple
                        if faces is ():
                            continue

                            # We iterate through our faces array and draw a rectangle
                            # over each face in faces
                        for (x,y,w,h) in faces:
                            cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2)

                        cv2.imwrite('/Users/amitsai/Desktop/capstone/yolo/darknet/data/1.jpg', frame)


                        result = face_match('/Users/amitsai/Desktop/capstone/yolo/darknet/data/1.jpg', 'data.pt')

                        print('Face matched with: ',result[0], 'With distance: ',result[1])
                        if result[0]=='U':
                            x=1
                            obj.set_value1(x)
                        else:
                            x=0
                            obj.set_value1(x)
                        
                    x1=0
                    if(class_id+1 == 77): 
                        print(scissors)
                        cv2.putText(frame, "scissors:detected", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                        x1=1
                        obj.set_value2(x1)
                    if(class_id+1 ==43): 
                        print(fork)
                        cv2.putText(frame, "fork:detected", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                        x1=1
                        obj.set_value2(x1)
                    if(class_id+1 ==44): 
                        print(knife)
                        cv2.putText(frame, "knife:detected", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                        x1=1
                        obj.set_value2(x1)
                    else:
                        obj.set_value2(x1)
            
        


        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)
    app.run(port=5000)