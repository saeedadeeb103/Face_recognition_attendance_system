import cv2 as cv
import face_recognition
import pickle
import os


# Importingg the Employees Images 
FolderPath = 'images'
PathList = os.listdir(FolderPath)
imgList = []
employesID = []

for path in PathList:
    imgList.append(cv.imread(os.path.join(FolderPath, path)))
    employesID.append(os.path.splitext(path)[0])

print(employesID)


def FindEncodeings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList
print("Encoding Started ....")
KnownEncodings = FindEncodeings(imgList)
KnownEncodingswithIDs = [KnownEncodings, employesID]
print("Encoding Compelete")

file = open("EncodeFile.p", 'wb')

pickle.dump(KnownEncodingswithIDs, file)
file.close()
print("File Saved")