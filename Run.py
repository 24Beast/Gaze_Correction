import numpy as np
from tensorflow.keras.models import load_model
import cv2
import dlib

class Pred:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("F:/Models/shape_predictor_68_face_landmarks.dat")
        self.model = load_model("F:/Projects/Beast/Models/L/L.h5")
    
    def get_eyes(self,frame):
        #Currently for a single face only
        faces = self.detector(frame)
        L_eye = -1
        R_eye = -1
        if(len(faces)==0):
            return -1,-1,-1,-1
        for face in faces: 
            shape = self.predictor(frame,face)
            L_max_x = shape.part(45).x + 10 if shape.part(45).x + 10>0 else 0
            L_min_x = shape.part(42).x - 10
            L_min_y = min([shape.part(43).y,shape.part(44).y]) - 10
            L_min_y = L_min_y if L_min_y > 0 else 0
            L_max_y = max([shape.part(47).y,shape.part(46).y]) + 10
            L_eye = np.reshape(cv2.resize(frame[L_min_y:L_max_y,L_min_x:L_max_x],(50,50)),(50,50,1))
            R_max_x = shape.part(39).x + 10 
            R_min_x = shape.part(36).x - 10 if shape.part(36).x - 10>0 else 0
            R_min_y = min([shape.part(37).y,shape.part(38).y]) - 10
            R_min_y = R_min_y if R_min_y > 0 else 0  
            R_max_y = max([shape.part(41).y,shape.part(40).y]) + 10
            R_eye = np.reshape(cv2.resize(frame[R_min_y:R_max_y,R_min_x:R_max_x],(50,50)),(50,50,1))
        return L_eye,R_eye,[L_min_y,L_max_y,L_min_x,L_max_x],[R_min_y,R_max_y,R_min_x,R_max_x]

    def get_pred(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        L_eye,R_eye,L_coords,R_coords = self.get_eyes(gray)
        if(type(L_eye)==int):
            return gray
        L_eye = np.reshape(L_eye,(50,50,1))/255
        R_eye = np.reshape(cv2.flip(R_eye,1),(50,50,1))/255
        L_eye,R_eye = self.model.predict(np.array([L_eye,R_eye]))
        L_eye = cv2.resize(L_eye,(L_coords[3]-L_coords[2],L_coords[1]-L_coords[0]))*255
        L_eye = L_eye.astype(np.uint8)
        R_eye = cv2.resize(cv2.flip(R_eye,1),(R_coords[3]-R_coords[2],R_coords[1]-R_coords[0]))*255
        R_eye = R_eye.astype(np.uint8)
        gray[L_coords[0]:L_coords[1],L_coords[2]:L_coords[3]] = L_eye
        gray[R_coords[0]:R_coords[1],R_coords[2]:R_coords[3]] = R_eye
        return gray

pred = Pred()

vid = cv2.VideoCapture(0)

while 1:
    ret,frame = vid.read()
    if(not(ret)):
        break
    new_frame = pred.get_pred(frame)
    cv2.imshow("Original",frame)
    cv2.imshow("New",new_frame)
    if cv2.waitKey(1) ==27:
        break
cv2.destroyAllWindows()
vid.release()
    
        