import cv2 as cv
import face_recognition
import pickle
import os

class EncodeGenerator:
    def __init__(self, image_folder):
        self.image_folder = image_folder

    def load_images(self):
        imgList = []
        employesID = []

        for filename in os.listdir(self.image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img_path = os.path.join(self.image_folder, filename)
                imgList.append(cv.imread(img_path))
                employesID.append(os.path.splitext(filename)[0])

        return imgList, employesID

    def find_encodings(self, imagesList):
        encodeList = []

        for img in imagesList:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0] if face_recognition.face_encodings(img) else None
            encodeList.append(encode)

        return encodeList

    def generate_and_save_encodings(self, output_file):
        imgList, employesID = self.load_images()
        print("Encoding Started ....")
        KnownEncodings = self.find_encodings(imgList)
        KnownEncodingswithIDs = [KnownEncodings, employesID]
        print("Encoding Complete")

        with open(output_file, 'wb') as file:
            pickle.dump(KnownEncodingswithIDs, file)

        print("File Saved")

if __name__ == "__main__":
    image_folder = 'images'
    output_file = 'EncodeFile.p'

    encoder = EncodeGenerator(image_folder)
    encoder.generate_and_save_encodings(output_file)
