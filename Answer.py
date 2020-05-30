# Importing Libraries
import numpy as np
import pickle
import cv2
import dlib
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Dense, Conv2D, Flatten, Reshape
from tensorflow.keras.models import Sequential

# Helper Functions
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("F:\Models\shape_predictor_68_face_landmarks.dat")

def get_eyes(frame):
    faces = detector(frame)
    L_eye = -1
    R_eye = -1
    for face in faces: 
        shape = predictor(gray,face)
        L_max_x = shape.part(45).x + 10 if shape.part(45).x + 10>0 else 0
        L_min_x = shape.part(42).x - 10
        L_min_y = min([shape.part(43).y,shape.part(44).y]) - 10
        L_min_y = L_min_y if L_min_y > 0 else 0
        L_max_y = max([shape.part(47).y,shape.part(46).y]) + 10
        print([L_min_y,L_max_y,L_min_x,L_max_x])
        L_eye = np.reshape(cv2.resize(frame[L_min_y:L_max_y,L_min_x:L_max_x],(50,50)),(50,50,1))
        R_max_x = shape.part(39).x + 10 
        R_min_x = shape.part(36).x - 10 if shape.part(36).x - 10>0 else 0
        R_min_y = min([shape.part(37).y,shape.part(38).y]) - 10
        R_min_y = R_min_y if R_min_y > 0 else 0  
        R_max_y = max([shape.part(41).y,shape.part(40).y]) + 10
        print([R_min_y,R_max_y,R_min_x,R_max_x])
        R_eye = np.reshape(cv2.resize(frame[R_min_y:R_max_y,R_min_x:R_max_x],(50,50)),(50,50,1))
    return L_eye,R_eye
        
# def get_closest(orig_frame,pred_frames):
#     index = np.argmin(pred_frames, key = lambda x: np.sum(np.absolute(x-orig_frame)))
#     return index

# Initializing data parameters
L_dict_nc = {}
L_dict_c = {}
R_dict_nc = {}
R_dict_c = {}
NC_data_dir = "F:/Datasets/NewGazeData/0/"
C_data_dir = "F:/Datasets/NewGazeData/1/"

# Getting Data
c_data_items = os.listdir(C_data_dir)
nc_data_items = os.listdir(NC_data_dir)

for item in c_data_items:
    img = cv2.imread(C_data_dir+item)
    name = "".join(item.split(".")[0].split("-")[0:-1])
    if(name not in L_dict_c):
        L_dict_c[name] = []
        R_dict_c[name] = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    L_eye,R_eye = get_eyes(gray)
    if(type(L_eye)==int):
        continue
    L_dict_c[name].append(L_eye)
    R_dict_c[name].append(R_eye)

for item in nc_data_items:
    img = cv2.imread(NC_data_dir+item)
    name = "".join(item.split(".")[0].split("-")[0:-1])
    if(name not in L_dict_nc):
        L_dict_nc[name] = []
        R_dict_nc[name] = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    L_eye,R_eye = get_eyes(gray)
    if(type(L_eye)==int):
        continue
    L_dict_nc[name].append(L_eye)
    R_dict_nc[name].append(R_eye)

pickle.dump(R_dict_nc,open("F:/Datasets/NewGazeData/R_nc.p","wb"))
pickle.dump(R_dict_c,open("F:/Datasets/NewGazeData/R_c.p","wb"))
pickle.dump(L_dict_nc,open("F:/Datasets/NewGazeData/L_nc.p","wb"))
pickle.dump(L_dict_c,open("F:/Datasets/NewGazeData/L_c.p","wb"))

R_X = []
for name in R_dict_nc.keys():
    if(len(R_dict_nc[name])==0):
        continue
    if(name in R_dict_c):
        if(len(R_dict_c[name])==0):
            continue
        pred_frames = R_dict_c[name]
        for orig_frame in R_dict_nc[name]:
            index_list = [np.sum(np.absolute(x-orig_frame)) for x in pred_frames]
            index = np.argmin(index_list)
            R_X.append([orig_frame,pred_frames[index]])

L_X = []
for name in L_dict_nc.keys():
    if(len(L_dict_nc[name])==0):
        continue
    if(name in L_dict_c):
        if(len(L_dict_c[name])==0):
            continue
        pred_frames = L_dict_c[name]
        for orig_frame in L_dict_nc[name]:
            index_list = [np.sum(np.absolute(x-orig_frame)) for x in pred_frames]
            index = np.argmin(index_list)
            L_X.append([orig_frame,pred_frames[index]])

R_X = np.array(R_X)
L_X = np.array(L_X)

np.save("F:/Datasets/NewGazeData/L_X",L_X)
np.save("F:/Datasets/NewGazeData/R_X",R_X)

L_y = np.reshape(L_X[:,1],(2006,50,50,1))/255
L_X = np.reshape(L_X[:,0],(2006,50,50,1))/255
R_y = np.reshape(R_X[:,1],(2006,50,50,1))/255
R_X = np.reshape(R_X[:,0],(2006,50,50,1))/255

# Create Left Eye Model
model_layers = [
    Conv2D(32,kernel_size=3,activation="relu",input_shape=(50,50,1)),
    Conv2D(64,kernel_size=3,activation="relu"),
    Conv2D(128,kernel_size=3,activation="relu"),
    Flatten(),
    Dense(64,activation="relu"),
    Dense(32,activation="tanh"),
    Dense(64,activation="relu"),
    Reshape((8,8,1)),
    Conv2DTranspose(128,kernel_size=3,activation="relu"),
    Conv2DTranspose(64,kernel_size=3,activation="relu"),
    Conv2DTranspose(32,kernel_size=3,activation="relu"),
    Flatten(),
    Dense(50*50,activation=None),
    Dense(50*50,activation="tanh"),
    Reshape((50,50,1))
    ]
L_model = Sequential(model_layers)
L_model.compile(loss="mean_absolute_error",optimizer="adam")
print(L_model.summary())

# Training Model
L_model.fit(L_X,L_y,batch_size=25,epochs=50)
#L_model.fit(right_X,right_y,batch_size=32,epochs=100)
pred = L_model.predict(L_X)

#Seeing Results
for i in range(0,25):
    cv2.imwrite("F:/Projects/Beast/Outputs/L/Original_"+str(i)+".jpg",cv2.resize(np.array(L_X[i]*255,dtype=np.uint8),(100,100)))
    cv2.imwrite("F:/Projects/Beast/Outputs/L/New_"+str(i)+".jpg",cv2.resize(np.array(pred[i]*255,dtype=np.uint8),(100,100)))

# Saving Model
L_model.save("F:/Projects/Beast/Models/L/L.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(L_model)
tflite_model = converter.convert()
open("F:/Projects/Beast/Models/L/L.tflite", "wb").write(tflite_model)


# Create Right Eye Model
model_layers = [
    Conv2D(32,kernel_size=3,activation="relu",input_shape=(50,50,1)),
    Conv2D(64,kernel_size=3,activation="relu"),
    Conv2D(128,kernel_size=3,activation="relu"),
    Flatten(),
    Dense(64,activation="relu"),
    Dense(32,activation="tanh"),
    Dense(64,activation="relu"),
    Reshape((8,8,1)),
    Conv2DTranspose(128,kernel_size=3,activation="relu"),
    Conv2DTranspose(64,kernel_size=3,activation="relu"),
    Conv2DTranspose(32,kernel_size=3,activation="relu"),
    Flatten(),
    Dense(50*50,activation=None),
    Dense(50*50,activation="tanh"),
    Reshape((50,50,1))
    ]
R_model = Sequential(model_layers)
R_model.compile(loss="mean_absolute_error",optimizer="adam")
print(L_model.summary())

# Training Model
R_model.fit(R_X,R_y,batch_size=25,epochs=50)
pred_R = R_model.predict(R_X)

#Seeing Results
for i in range(0,25):
    cv2.imwrite("F:/Projects/Beast/Outputs/R/Original_"+str(i)+".jpg",cv2.resize(np.array(R_X[i]*255,dtype=np.uint8),(100,100)))
    cv2.imwrite("F:/Projects/Beast/Outputs/R/New_"+str(i)+".jpg",cv2.resize(np.array(pred_R[i]*255,dtype=np.uint8),(100,100)))

# Saving Model
L_model.save("F:/Projects/Beast/Models/R/R.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(R_model)
tflite_model = converter.convert()
open("F:/Projects/Beast/Models/R/R.tflite", "wb").write(tflite_model)

