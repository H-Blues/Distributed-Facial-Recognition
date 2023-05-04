# Distributed-Facial-Recognition
A simple distributed web application based on Flask, to detect the known faces. 

This is an CA for course Distributed Systems.

The demo of this project can be accessible through YouTube: https://youtu.be/H0QXWC5jQ7E

## Overview

To achieve facial recognition, I utilized OpenCV and face-recognition libraries to identify faces immediately. This involved using face_location() to locate faces and face_encodings() to encode the facial features as 128-dimensional vectors, thus being able to compare and match faces.

For the data transmission component, I used Flask-SocketIO to create a real-time bidirectional connection between the server and clients. When a client connects to the server, a new thread is created to receive and process image data from the client. To prevent disruption to the main thread, I used a queue system to store incoming image data and process it in a separate thread.

In summary, the biggest advantage of my facial recognition system is that it allows different devices to access the webpage and call their own cameras to transmit different recognition results. At the same time, I have also implemented compression processing during the image transmission process. However, the low frame rate transmission results in an unsmooth picture, and considering the issues of security and encryption in data transmission is also a direction for future improvement.

## References

- Krish Naik. (2020, July 7). Face Recognition Project In Flask Web Framework- Recognize Different Faces Easily. [Video]. YouTube. https://www.youtube.com/watch?v=Az1MH_e1hVA
-	Miguel Grinberg. (2018). Flask-SocketIO documentation. Flask-SocketIO. https://flask-socketio.readthedocs.io/en/latest/
-	MrSpeedy68. Live-Streaming-Application. GitHub. https://github.com/MrSpeedy68/Live-Streaming-Application
-	ageitgey. (n.d.). face_recognition. GitHub. https://github.com/ageitgey/face_recognition

