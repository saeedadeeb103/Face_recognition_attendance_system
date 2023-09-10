import cv2 as cv
import os
import pickle
import face_recognition
import numpy as np
import cvzone
from firebase_admin import db
from DataManager import DataManager 

class FaceRecognitionAttendanceSystem:
    def __init__(self):
        self.data_manager = DataManager()
        self.imgsz = (640, 480)
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.cap.set(3, self.imgsz[0])
        self.cap.set(4, self.imgsz[1])
        self.counter = 0 
        self.modeType = 0
        self.ID = -1
        self.imgBackground = cv.imread('resources/background.png')
        self.FolderModePath = 'resources/Modes'
        self.ModePathList = os.listdir(self.FolderModePath)
        self.imgModeList = [cv.imread(os.path.join(self.FolderModePath, path)) for path in self.ModePathList]

        self.load_encoded_data()
    
    def load_encoded_data(self):
        print("Loading Encoded File .......")
        with open('EncodeFile.p', 'rb') as file:
            KnownEncodingwithIDs = pickle.load(file)

        self.KnownEncodings, self.employesID = KnownEncodingwithIDs
        print("Encoded File Loaded")

    def mark_attendance(self):
        while True:
            success, img = self.cap.read()
            imgSize = cv.resize(img, (0, 0), None, 0.25, 0.25)
            imgSize = cv.cvtColor(imgSize, cv.COLOR_BGR2RGB)

            faceCurrentFrame = face_recognition.face_locations(imgSize)
            encodeCurrentFrame = face_recognition.face_encodings(imgSize, faceCurrentFrame)
            self.imgBackground[162:162 + 480, 55:55 + 640] = img
            self.imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[0]

            for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
                matches = face_recognition.compare_faces(self.KnownEncodings, encodeFace)
                Face_Dis = face_recognition.face_distance(self.KnownEncodings, encodeFace)
                matchIndex = np.argmin(Face_Dis)

                if matches[matchIndex]:
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                    self.imgBackground = cvzone.cornerRect(self.imgBackground, bbox, rt=0)
                    self.ID = self.employesID[matchIndex]
                    if self.counter ==0: 
                        self.counter = 1

            if self.counter != 0: 

                if self.counter ==1: 
                    employee_info = self.data_manager.get_employee_info_by_id(self.ID)
                    print(employee_info)
                self.counter+= 1

            cv.imshow("Face Attendance", self.imgBackground)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    def run(self):
        self.mark_attendance()
        self.cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    attendance_system = FaceRecognitionAttendanceSystem()
    attendance_system.run()
