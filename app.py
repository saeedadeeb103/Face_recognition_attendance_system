from flask import Flask, render_template, Response, redirect, url_for, send_from_directory
import cv2
import face_recognition
from main import FaceRecognitionAttendanceSystem
from DataManager import DataManager
from datetime import datetime
import cvzone
import time
app = Flask(__name__, template_folder='./templates')

imgsz = (640, 480)
camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
attendance_system = FaceRecognitionAttendanceSystem()
date = datetime.now().strftime("%Y-%m-%d")
employee_info = {}

def gen_frames():
    global date, employee_info, recognized_face_detected
    recognized_face_detected = False
    while True:
        success, frame = camera.read()
        if success:
            imgSize = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgSize = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faceCurrentFrame = face_recognition.face_locations(imgSize)
            
            employee_info = {}
            employee_ids = []
            for faceLoc in faceCurrentFrame:
                face_encoding = face_recognition.face_encodings(imgSize, [faceLoc])[0]
                matches = face_recognition.compare_faces(attendance_system.KnownEncodings, face_encoding)
                if any(matches):
                    employee_id = attendance_system.employesID[matches.index(True)]

                    employee_info = attendance_system.data_manager.get_employee_info_by_id(employee_id=employee_id)
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