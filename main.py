import cv2 as cv 
import os
import pickle
import face_recognition
import numpy as np 
import cvzone
imgsz = (640, 480)
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(3, imgsz[0])
cap.set(4, imgsz[1])


# Importingg the Modes Images into a List 
imgBackground = cv.imread('resources/background.png')
FolderModePath = 'resources/Modes'
ModePathList = os.listdir(FolderModePath)
imgModeList = []
for path in ModePathList:
    imgModeList.append(cv.imread(os.path.join(FolderModePath, path)))

# Load the encoding file 
print("Loading Encoded File .......")
file = open('EncodeFile.p', 'rb')
KnownEncodingwithIDs  = pickle.load(file)
file.close()

KnownEncodings, employesID = KnownEncodingwithIDs
# print(employesID)
print("Encoded File Loaded")
while True:
    success, img = cap.read()

    imgSize = cv.resize(img, (0,0), None, 0.25, 0.25)
    imgSize = cv.cvtColor(imgSize, cv.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(imgSize)
    encodeCurrentFrame = face_recognition.face_encodings(imgSize, faceCurrentFrame)
    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[44:44+633, 808:808+414] = imgModeList[1]

    for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(KnownEncodings, encodeFace)
        Face_Dis = face_recognition.face_distance(KnownEncodings, encodeFace)
        matchIndex = np.argmin(Face_Dis)

        if matches[matchIndex]:
            # print(employesID[matchIndex])
            # print("Known Faces Detected.")
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            bbox = 55+x1, 162+y1, x2-x1, y2-y1 
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt= 0 )
    cv.imshow("Face Attendance", imgBackground)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()