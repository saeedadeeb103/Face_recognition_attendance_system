import firebase_admin
from firebase_admin import credentials
from firebase_admin import db 
from firebase_admin import storage
import os 
import json

class DataManager:
    def __init__(self):
        self.cred = credentials.Certificate("service_account_key.json")
        firebase_admin.initialize_app(self.cred, {
            "databaseURL": "https://employer-tracker-6c2be-default-rtdb.firebaseio.com/",
            "storageBucket": "employer-tracker-6c2be.appspot.com",
        })
        self.ref = db.reference("employers")

    def upload_employee_data(self, data):
        for key, value in data.items():
            self.ref.child(key).set(value)
        print("Employee data uploaded to the Realtime Database.")

    def upload_images_to_storage(self, image_paths):
        storage_ref = storage.bucket()

        for path in image_paths:
            # Construct a unique storage path for each image (e.g., "images/395.jpg")
            image_filename = os.path.basename(path)
            storage_path = f"images/{image_filename}"

            # Upload the image to Firebase Storage
            blob = storage_ref.blob(storage_path)
            blob.upload_from_filename(path)

            # Get the public URL of the uploaded image
            image_url = blob.public_url
            print(f"Uploaded image: {image_url}")
        
        print("Images uploaded to Firebase Storage.")

    def load_employee_data_from_json(self, json_file):
        with open(json_file, 'r') as file:
            employee_data = json.load(file)
        return employee_data

if __name__ == "__main__":
    data_manager = DataManager()

    # Load employee data from the JSON file
    employee_data = data_manager.load_employee_data_from_json('data/employee_data.json')

    # Upload employee data to the Realtime Database
    data_manager.upload_employee_data(employee_data)

    # Add 'images/' prefix to PathList
    image_folder = 'images/'
    
    image_paths = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img_path = os.path.join(image_folder, filename)
            image_paths.append(img_path)

    # Upload images to Firebase Storage
    data_manager.upload_images_to_storage(image_paths)
