import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import os
import json
import numpy as np
import cv2 as cv
from datetime import datetime
from encodeGen import EncodeGenerator
image_folder = 'images'
output_file = 'EncodeFile.p'

class DataManager:
    def __init__(self):
        self.cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(self.cred, {
            "databaseURL": "https://employer-tracker-6c2be-default-rtdb.firebaseio.com/",
            "storageBucket": "employer-tracker-6c2be.appspot.com",
        })
        self.ref = db.reference("Data")
        self.storage_bucket = storage.bucket()

    def upload_employee_data(self, data):
        employees_ref = self.ref

        # Get the current employee IDs from the Realtime Database
        current_ids = set(employees_ref.child("employees").get().keys()) if employees_ref.get() else set()
        import pdb
        
        for key, value in data['employees'].items():
            # Check if the employee ID already exists in the database
            
            if key not in current_ids:
                
                # Add the employee data since it doesn't exist
                employees_ref.child("employees").child(key).set(value)
                print(f"Employee data for ID {key} added to the Realtime Database.")
            else:
                print(f"Employee with ID {key} already exists in the database. Skipping.")


    def upload_images_to_storage(self, image_paths):
        storage_ref = storage.bucket()
        
        # Get the list of image filenames already in Firebase Storage
        current_image_filenames = [blob.name.split("/")[-1] for blob in storage_ref.list_blobs()]

        for path in image_paths:
            # Extract the filename from the local path
            image_filename = os.path.basename(path)
            image = cv.imread(path)

            # Check if the image size is not 216x216 pixels
            if image.shape[0] != 216 or image.shape[1] != 216:
                # Resize the image to 216x216 pixels
                image = cv.resize(image, (216, 216))

            # Check if the image filename already exists in Firebase Storage
            if image_filename not in current_image_filenames:
                # Construct a unique storage path for the image
                storage_path = f"images/{image_filename}"

                # Upload the image to Firebase Storage
                blob = storage_ref.blob(storage_path)
                blob.upload_from_string(cv.imencode('.jpg', image)[1].tostring(), content_type='image/jpeg')

                # Get the public URL of the uploaded image
                image_url = blob.public_url
                print(f"Uploaded image: {image_url}")
                encoder = EncodeGenerator(image_folder=image_folder)
                encoder.generate_and_save_encodings(output_file=output_file)
            else:
                print(f"Image {image_filename} already exists in Firebase Storage. Skipping.")

        print("Images uploaded to Firebase Storage.")

    def load_employee_data_from_json(self, json_file):
        with open(json_file, 'r') as file:
            employee_data = json.load(file)
        return employee_data

    def get_employee_info_by_id(self, employee_id):
        # Navigate to the "employees" node and search for the employee by ID
        employees_ref = self.ref.child("employees")
        employee_data = employees_ref.child(employee_id).get()
        return employee_data

    def get_employee_image_by_id(self, employee_id):
        # Construct the storage path for the employee's image
        storage_path = f"images/{employee_id}.jpg"

        # Get the download URL of the image
        blob = self.storage_bucket.get_blob(storage_path)
        array = np.frombuffer(blob.download_as_string(), np.uint8)
        employee_img = cv.imdecode(array, cv.COLOR_BGRA2BGR)
        return employee_img

    def update_employee_login_logout_time(self, employee_id, login_logout_time):
        # Navigate to the "employees" node and search for the employee by ID
        employees_ref = self.ref.child("employees")
        employee_data = employees_ref.child(employee_id).get()

        if employee_data:
            # Check if there is "attendance" data for the employee
            attendance = employee_data.get("attendance", {})
            new_date = login_logout_time.strftime("%Y-%m-%d")  # Convert datetime to string with format

            # Check if there's already an entry for the same day
            if new_date not in attendance:
                attendance[new_date] = {"login_time": login_logout_time.strftime("%H:%M:%S"), "logout_time": None, "delays": 0}
                expected = datetime(login_logout_time.year , login_logout_time.month, login_logout_time.day, 8, 0 , 0)
                if login_logout_time > expected: 
                    delay = login_logout_time - expected
                    delay_seconds = delay.total_seconds()
                if delay_seconds > 0: 
                    attendance[new_date]["delays"] = int(delay_seconds)
                else: 
                    attendance[new_date]["delays"] = 0

            else:
                # If there's already an entry, it means the employee logged out later on the same day
                attendance[new_date]["logout_time"] = login_logout_time.strftime("%H:%M:%S")
            
            

            employees_ref.child(employee_id).child("attendance").set(attendance)

    def is_same_date(self, date1, date2):
        # Check if two datetime strings represent the same date
        date_format = "%Y-%m-%d"
        date2_str = date2.strftime(date_format)  # Format date2 as a string in the same format
        return date1[:10] == date2_str

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

    # Example: Get employee info by ID (e.g., "395")