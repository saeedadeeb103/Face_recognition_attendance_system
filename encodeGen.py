import cv2 as cv
import face_recognition
import pickle
from PIL import Image, ImageEnhance
import os
import numpy as np

class EncodeGenerator:
    def __init__(self, image_folder):
        self.image_folder = image_folder

    def enhance_image_quality(self, img, contrast_factor=1.5, brightness_factor=1.2):
        # Open the image using PIL
        pil_img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(contrast_factor)

        # Enhance brightness
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(brightness_factor)

        # Convert the enhanced image back to OpenCV format
        enhanced_img = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)

        return enhanced_img
    
    def display_enhanced_image(self, img):
        enhanced_img = self.enhance_image_quality(img)
        cv.imshow('Enhanced Image', enhanced_img)
        cv.waitKey(0)
        cv.destroyAllWindows()


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
            enhanced_img = self.enhance_image_quality(img)
            encode = face_recognition.face_encodings(enhanced_img)[0] if face_recognition.face_encodings(enhanced_img) else None
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
    imgList, employesID = encoder.load_images()

    # Display the enhanced image for the first employee
    # enhanced_img = encoder.enhance_image_quality(imgList[2], contrast_factor=1.5, brightness_factor=1.2)
    # cv.imshow('Enhanced Image', enhanced_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()