import cv2 as cv
import os
import pickle
import face_recognition
import numpy as np
import cvzone
from firebase_admin import db
from DataManager import DataManager 
from datetime import datetime

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
            # print("Face_ Location ", faceCurrentFrame)
            encodeCurrentFrame = face_recognition.face_encodings(imgSize, faceCurrentFrame)
            # print("Face Encode Current Frame", encodeCurrentFrame)
            self.imgBackground[162:162 + 480, 55:55 + 640] = img
            self.imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[self.modeType]

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
                    if self.counter == 0:
                        self.counter = 1
                        self.modeType = 1
                        self.record_login_time()  # Record login time on first scan
                    else:
                        self.record_logout_time()  # Record logout time on subsequent scans

            if self.counter != 0: 
                if self.counter == 1: 
                    # Get Data 
                    employee_info = self.data_manager.get_employee_info_by_id(self.ID)

                    # Get The Image from the Storage 
                    employee_img = self.data_manager.get_employee_image_by_id(self.ID)

                if 10 < self.counter < 20: 
                    self.modeType = 2

                self.imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[self.modeType]
                if self.counter <= 10:
                    #import pdb 
                    #pdb.set_trace()
                    date = datetime.now()
                    date = date.strftime("%Y-%m-%d")
                    cv.putText(self.imgBackground, str(employee_info['attendance'][date]['login_time']), (861, 125), 
                            cv.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1)
                    cv.putText(self.imgBackground, str(employee_info['position']), (1006, 550), 
                            cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
                    cv.putText(self.imgBackground, str(self.ID), (1006, 493), 
                            cv.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv.putText(self.imgBackground, str(employee_info['attendance'][date]['delays']), (910, 625), 
                            cv.FONT_HERSHEY_COMPLEX, 0.5, (100, 100, 100), 1)
                    cv.putText(self.imgBackground, str(employee_info['starting_year']), (1125, 625), 
                            cv.FONT_HERSHEY_COMPLEX, 0.5, (100, 100, 100), 1)

                    (w,h ), _ = cv.getTextSize(employee_info['name'], cv.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414-w)//2
                    cv.putText(self.imgBackground, str(employee_info['name']), (808+offset, 455), 
                            cv.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                    self.imgBackground[175:175+216, 909:909+216] = employee_img
                self.counter+= 1

                if self.counter >= 20: 
                    self.counter = 0 
                    self.modeType = 0 
                    employee_info = []
                    employee_img = []
                    self.imgBackground[44:44 + 633, 808:808 + 414] = self.imgModeList[self.modeType]

            cv.imshow("Face Attendance", self.imgBackground)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    def record_login_time(self):
        # Record login time for the employee in Firebase
        current_time = datetime.now()
        self.data_manager.update_employee_login_logout_time(self.ID, current_time)

    def record_logout_time(self):
        # Record logout time for the employee in Firebase
        current_time = datetime.now()
        self.data_manager.update_employee_login_logout_time(self.ID, current_time)

    def run(self):
        self.mark_attendance()
        self.cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    attendance_system = FaceRecognitionAttendanceSystem()
    attendance_system.run()
