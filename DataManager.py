import firebase_admin
from firebase_admin import credentials
from firebase_admin import db 
from firebase_admin import storage
import os 
import json
import numpy as np
import cv2 as cv 
from datetime import datetime

class DataManager:
    def __init__(self):
        self.cred = credentials.Certificate("service_account_key.json")
        firebase_admin.initialize_app(self.cred, {
            "databaseURL": "https://employer-tracker-6c2be-default-rtdb.firebaseio.com/",
            "storageBucket": "employer-tracker-6c2be.appspot.com",
        })
        self.ref = db.reference("Data")
        self.storage_bucket = storage.bucket()

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

    def get_employee_info_by_id(self, employee_id):
        # Navigate to the "employees" node and search for the employee by ID
        employees_ref = self.ref.get()
        for employee in employees_ref.get("employees", []):
            if employee.get("id") == employee_id:
                return employee
        print(employee)
        # If no match is found, return None
        return None


    
    def get_employee_image_by_id(self, employee_id):
        # Construct the storage path for the employee's image
        storage_path = f"images/{employee_id}.jpg"

        # Get the download URL of the image
        blob = self.storage_bucket.get_blob(storage_path)
        array = np.frombuffer(blob.download_as_string(), np.uint8)
        employee_img = cv.imdecode(array, cv.COLOR_BGRA2BGR)
        return employee_img

    def update_employee_login_time(self, employee_id, login_time):
        # Get the current employee data
        employee_data = self.get_employee_info_by_id(employee_id)
        
        # Check if there is "attendance" data for the employee
        if "attendance" in employee_data:
            # Iterate through attendance records to find the right date
            for record in employee_data["attendance"]:
                if self.is_same_date(record["date"], login_time):
                    # Update the login time for the specified date
                    record["login_time"] = login_time
                    self.ref.child("employees").child(employee_id).update(employee_data)
                    break

    def is_same_date(self, date1, date2):
        # Check if two datetime strings represent the same date
        date_format = "%Y-%m-%d"
        return datetime.strptime(date1[:10], date_format) == datetime.strptime(date2[:10], date_format)

    def update_employee_logout_time(self, employee_id, logout_time):
        # Get the current employee data
        employee_data = self.ref.child("employees").child(employee_id).get()
        
        # Check if there is "attendance" data for the employee
        if "attendance" in employee_data:
            # Iterate through attendance records to find the right date
            for record in employee_data["attendance"]:
                if self.is_same_date(record["date"], logout_time):
                    # Update the logout time for the specified date
                    record["logout_time"] = logout_time
                    self.ref.child("employees").child(employee_id).update(employee_data)
                    break

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
