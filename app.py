from flask import Flask, render_template, request
from flask_socketio import SocketIO
import cv2
from FaceRecognizer import FaceRecognizer
from threading import Thread
import base64

app = Flask(__name__)
socketio = SocketIO(app)

stop_camera_feed = False

# Initialize the face recognizer with known faces
face_recognizer = FaceRecognizer('./known_faces.txt')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    # Create a new thread for each client to handle the camera feed
    thread = Thread(target=handle_camera_feed)
    thread.start()

@socketio.on('disconnect')
def handle_disconnect():
    # Set a flag to stop the camera feed thread
    global stop_camera_feed
    stop_camera_feed = True
    print("disconnect")

def handle_camera_feed():
    global stop_camera_feed
    stop_camera_feed = False

    # Initialize the camera
    camera = cv2.VideoCapture(0)

    while not stop_camera_feed:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Recognize faces in the current frame
            image_bytes = face_recognizer.recognize_faces(frame)
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            # Send the image bytes to the client
            socketio.emit('image', image_base64)

    # Release the camera when we're done
    camera.release()


if __name__ == '__main__':
    socketio.run(app, debug=True)
