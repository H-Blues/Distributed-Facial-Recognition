import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO
import cv2
from FaceRecognizer import FaceRecognizer
from threading import Thread
import base64

app = Flask(__name__)
socketio = SocketIO(app)
connections = {}
data_uri_queue = {}

# Initialize the face recognizer with known faces
face_recognizer = FaceRecognizer('./known_faces.txt')

def handle_camera_feed(sid):
    while sid in connections:
        if len(data_uri_queue[sid]) > 0:
            data_uri = data_uri_queue[sid].pop(0)
            handle_origin_img(sid, data_uri)

def handle_origin_img(sid, data_uri):
    data_bytes = base64.b64decode(data_uri.split(',')[1])
    np_arr = np.frombuffer(data_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_bytes = face_recognizer.recognize_faces(frame)
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    socketio.emit('detectedImage', image_base64, room=sid)

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    connections[sid] = True
    data_uri_queue[sid] = []
    thread = Thread(target=handle_camera_feed, args=(sid,))
    thread.start()

@socketio.on('originImage')
def handle_origin_img_event(data_uri):
    sid = request.sid
    data_uri_queue[sid].append(data_uri)

@socketio.on('disconnect')
def handle_disconnect():
    # Remove the disconnected client's connection from our dictionary
    connections.pop(request.sid, None)
    print("disconnect: " + str(request.sid))

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)
