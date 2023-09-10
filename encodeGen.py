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
    path
