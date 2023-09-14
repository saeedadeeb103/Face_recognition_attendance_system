from flask import Flask, render_template, Response, request, jsonify
import cv2
import datetime
import os
import cvzone
import numpy as np
from threading import Thread
import face_recognition
from main import FaceRecognitionAttendanceSystem

# Global variables for controlling features
capture = 0
grey = 0
neg = 0
face = 0
rec = 0
face_detection_enabled = False  # To enable/disable face detection

# Make 'shots' directory to save pictures
try:
    os.makedirs('./shots', exist_ok=True)
except OSError as error:
    pass

# Instantiate Flask app
app = Flask(__name__, template_folder='./templates')
imgsz = (640, 480)
camera = cv2.VideoCapture(0)
camera.set(3, imgsz[0])
camera.set(4, imgsz[1])
face_detection_enabled = False
attendance_system = FaceRecognitionAttendanceSystem()

def record(out):
    global rec_frame
    while rec:
        time.sleep(0.05)
        out.write(rec_frame)

def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if grey:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if neg:
                frame = cv2.bitwise_not(frame)
            if capture:
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)
            if rec:
                rec_frame = frame
                frame = cv2.putText(cv2.flip(frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                   4)
                frame = cv2.flip(frame, 1)

            if face_detection_enabled:
                # Perform face detection and draw rectangles around detected faces
                import pdb
                imgSize = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
                imgSize = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faceCurrentFrame = face_recognition.face_locations(imgSize)
                # pdb.set_trace()
                employee_ids = []
                for faceLoc in faceCurrentFrame:
                    # Perform face recognition on the detected face
                    face_encoding = face_recognition.face_encodings(imgSize, [faceLoc])[0]
                    matches = face_recognition.compare_faces(attendance_system.KnownEncodings, face_encoding)
                    if any(matches):
                        # Face belongs to an employee
                        employee_id = attendance_system.employesID[matches.index(True)]
                        employee_ids.append(employee_id)  # You can use True to represent that the face belongs to an employee
                    else:
                        # Face doesn't belong to an employee
                        employee_ids.append(None)  # You can use False to represent that the face doesn't belong to an employee

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
            except Exception as e:
                pass

        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global capture, grey, neg, face, rec, out, face_detection_enabled
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            capture = 1
        elif request.form.get('grey') == 'Grey':
            grey = not grey
        elif request.form.get('neg') == 'Negative':
            neg = not neg
        elif request.form.get('face') == 'Face Only':
            # Toggle face detection
            face_detection_enabled = not face_detection_enabled
        elif request.form.get('stop') == 'Stop/Start':
            if rec:
                rec = 0
                out.release()
            else:
                rec = 1
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":", '')), fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run()

# Release the camera and close OpenCV windows when the Flask app is stopped
@app.after_request
def release_camera(response):
    camera.release()
    cv2.destroyAllWindows()
    return response
