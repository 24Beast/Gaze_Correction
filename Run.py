import numpy as np
from tensorflow.keras.models import load_model
import cv2
import dlib

class Pred:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("F:/Models/shape_predictor_68_face_landmarks.dat")
        self.model = load_model("F:/Projects/Beast/Models/L/L.h5")
        self.kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        print("Models Initialized")
    
    def sharpen(self,frame):
        sharp_frame = cv2.filter2D(frame, -1, self.kernel)
        return sharp_frame
        
    def get_eyes(self,frame):
        #Currently for a single face only
        faces = self.detector(frame)
        L_eye = -1
        R_eye = -1
        L_mask = np.zeros((50,50))
        R_mask = np.zeros((50,50))
        if(len(faces)==0):
            return -1,-1,-1,-1,-1,-1
        for face in faces: 
            shape = self.predictor(frame,face)
            L_max_x = shape.part(45).x + 10 if shape.part(45).x + 10>0 else 0
            L_min_x = shape.part(42).x - 10
            L_min_y = min([shape.part(43).y,shape.part(44).y]) - 10
            L_min_y = L_min_y if L_min_y > 0 else 0
            L_max_y = max([shape.part(47).y,shape.part(46).y]) + 10
            L_eye = np.reshape(cv2.resize(frame[L_min_y:L_max_y,L_min_x:L_max_x],(50,50)),(50,50,1))
            L_roi_corners = np.array([[(shape.part(i).x-L_min_x,shape.part(i).y-L_min_y) for i in range(42,48)]])
            L_mask = cv2.fillPoly(np.zeros((L_max_y-L_min_y,L_max_x-L_min_x)),L_roi_corners,(255))
            R_max_x = shape.part(39).x + 10 
            R_min_x = shape.part(36).x - 10 if shape.part(36).x -   10>0 else 0
            R_min_y = min([shape.part(37).y,shape.part(38).y]) - 10
            R_min_y = R_min_y if R_min_y > 0 else 0
            R_max_y = max([shape.part(41).y,shape.part(40).y]) + 10
            R_roi_corners = np.array([[(shape.part(i).x-R_min_x,shape.part(i).y-R_min_y) for i in range(36,42)]])
            R_mask = cv2.fillPoly(np.zeros((R_max_y-R_min_y,R_max_x-R_min_x)),R_roi_corners,(255))
            R_eye = np.reshape(cv2.resize(frame[R_min_y:R_max_y,R_min_x:R_max_x],(50,50)),(50,50,1))
            for i in range(10):
                L_eye = self.sharpen(L_eye)
                R_eye = self.sharpen(R_eye)
        return L_eye,R_eye,[L_min_y,L_max_y,L_min_x,L_max_x],[R_min_y,R_max_y,R_min_x,R_max_x],L_mask.astype(np.uint8),R_mask.astype(np.uint8)

    def get_pred(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        L_eye,R_eye,L_coords,R_coords,L_mask,R_mask = self.get_eyes(gray)
        if(type(L_eye)==int):
            return frame
        L_eye = np.reshape(L_eye,(50,50,1))/255
        R_eye = np.reshape(cv2.flip(R_eye,1),(50,50,1))/255
        L_eye,R_eye = self.model.predict(np.array([L_eye,R_eye]))
        L_eye = cv2.resize(L_eye,(L_coords[3]-L_coords[2],L_coords[1]-L_coords[0]))*255
        L_eye = L_eye.astype(np.uint8)
        R_eye = cv2.resize(cv2.flip(R_eye,1),(R_coords[3]-R_coords[2],R_coords[1]-R_coords[0]))*255
        R_eye = R_eye.astype(np.uint8)
        L_eye_masked = cv2.cvtColor(cv2.bitwise_and(L_eye, L_mask),cv2.COLOR_GRAY2BGR)
        R_eye_masked = cv2.cvtColor(cv2.bitwise_and(R_eye, R_mask),cv2.COLOR_GRAY2BGR)
        L_orig_masked = cv2.bitwise_and(frame[L_coords[0]:L_coords[1],L_coords[2]:L_coords[3]].astype(np.uint8),cv2.cvtColor(255-L_mask,cv2.COLOR_GRAY2BGR))
        R_orig_masked = cv2.bitwise_and(frame[R_coords[0]:R_coords[1],R_coords[2]:R_coords[3]].astype(np.uint8),cv2.cvtColor(255-R_mask,cv2.COLOR_GRAY2BGR))
        frame[L_coords[0]:L_coords[1],L_coords[2]:L_coords[3]] = L_orig_masked + L_eye_masked
        frame[R_coords[0]:R_coords[1],R_coords[2]:R_coords[3]] = R_orig_masked + R_eye_masked
        return frame

pred = Pred()
vid = cv2.VideoCapture("rec2.mp4")
out = cv2.VideoWriter('F:/Projects/Beast/outpy_4.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480))
print("Starting Recording")
while 1:
    ret,img = vid.read()
    if(not(ret)):
        break
    img = cv2.resize(img,(640,480))
    new_frame = pred.get_pred(img)
    cv2.imshow("Original",img)
    cv2.imshow("New",new_frame)
    out.write(new_frame)
    if cv2.waitKey(1) ==27:
        break
cv2.destroyAllWindows()
vid.release()
out.release()