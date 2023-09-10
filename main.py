import cv2 as cv 
import os

imgsz = (640, 480)
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(3, imgsz[0])
cap.set(4, imgsz[1])


# Importingg the Modes Images into a List 
imgBackground = cv.imread('resources/background.png')
FolderModePath = 'resources/Modes'
ModePathList = os.listdir(FolderModePath)
imgModeList = []

print(ModePathList)
for path in ModePathList:
    imgModeList.append(cv.imread(os.path.join(FolderModePath, path)))
print(len(imgModeList))

while True:
    success, img = cap.read()
    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[44:44+633, 808:808+414] = imgModeList[1]
    cv.imshow("Face Attendance", imgBackground)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()