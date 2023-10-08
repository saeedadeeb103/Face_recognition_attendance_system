from flask import Flask, render_template, Response, redirect, url_for, send_from_directory
import cv2
import face_recognition
from main import FaceRecognitionAttendanceSystem
from DataManager import DataManager
from datetime import datetime
import cvzone
import numpy as np 

app = Flask(__name__, template_folder='./templates')

imgsz = (640, 480)
camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
attendance_system = FaceRecognitionAttendanceSystem()
date = datetime.now().strftime("%Y-%m-%d")
employee_info = {}
id = -1
counter = 0
def gen_frames():
    global date, employee_info, recognized_face_detected
    recognized_face_detected = False
    counter = 0
    while True:
        success, frame = camera.read()
        if success:
            imgSize = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgSize = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faceCurrentFrame = face_recognition.face_locations(imgSize)
            encodeCurrentFrame = face_recognition.face_encodings(imgSize, faceCurrentFrame)
            
            employee_info = {}
            employee_ids = []
            for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
                #face_encoding = face_recognition.face_encodings(imgSize, [faceLoc])[0]
                matches = face_recognition.compare_faces(attendance_system.KnownEncodings, encodeFace)
                face_distance = face_recognition.face_distance(attendance_system.KnownEncodings, encodeFace)
                matcheIndex = np.argmin(face_distance)

                if matches[matcheIndex]:
                    employee_id = attendance_system.employesID[matches.index(True)]

                    #employee_info = attendance_system.data_manager.get_employee_info_by_id(employee_id=employee_id)
                    #print(employee_info)
                    employee_ids.append(employee_id)
                    recognized_face_detected = True
                    break
                else: 
                    employee_ids.append(None)   
            for (top, right, bottom, left), employee_id in zip(faceCurrentFrame, employee_ids):
                if employee_id is not None:
                    # This face belongs to an employee
                    bbox = (left, top, right - left, bottom - top)
                    frame = cvzone.cornerRect(frame, bbox, rt=0, colorC=(0, 255, 0))
                    # counter is used as timer to check if there is employee recorded show the employee data for certain sec and then returns it detected (works as counter between the modes)
                    # first mode is to show that the system is active 
                    # second mode is to show the employee data from image and other info 
                    # third is to show that the employee attendance has been recorded
                    if counter == 0:
                        counter =1 
                        mode = 1 
                        print("The data of the emplpyee is shown  ", mode)
                        current_time = datetime.now()
                        expected = datetime(current_time.year, current_time.month, current_time.day, 9, 0, 0)
                        if current_time > expected:
                            delay = current_time - expected 
                            delay_sec = delay.total_seconds()
                        else:
                            delay_sec = 0
                        attendance_system.data_manager.update_employee_login_logout_time(employee_id, current_time)
                    else:
                        attendance_system.data_manager.update_employee_login_logout_time(employee_id, current_time)

            if counter != 0:
                if counter ==1:
                    employee_info = attendance_system.data_manager.get_employee_info_by_id(employee_id)
                    # get the image of the employee so can redirected to the detect and show it in the employee info 
                    employee_img = attendance_system.data_manager.get_employee_image_by_id(employee_id)

                if 10 < counter < 20:
                    mode = 1 # to show the data of the employee mode 
                    print("The data of the emplpyee is shown  ", mode)
                if counter <= 10:
                    print("The data of the emplpyee is being shown")
                
                counter += 1
                
                if counter >= 20:
                    mode = 2
                    print("The attendance is recorded", mode)
                if counter >= 25:
                    counter = 0
                    mode = 0 # to show that system is active
                    print("The System is active",mode)
                    employee_info = {}
                    employee_img = []
                    employee_ids = []

            try: 
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                if recognized_face_detected:
                    # Redirect to the 'detect' route with employee_info
                    return redirect(url_for('detect'))
            except:
                pass
        else:
            pass

@app.route('/employee_info')
def employee_info():
    return render_template("employee_info.html", employee_info=employee_info)

@app.route('/')
def index():
    return render_template('index.html', date=date, employee_info=employee_info)

@app.route('/detect')
def detect():
    print(employee_info)
    return render_template('detect.html', employee_info=employee_info)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()


@app.after_request
def release_camera(response):
    camera.release()
    cv2.destroyAllWindows()
    return response
