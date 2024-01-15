from flask import Flask, Response
import cv2
import pickle
from keras.models import load_model
from keras.utils import img_to_array
import imutils
from PIL import Image as im
import numpy as np

app = Flask(__name__)
video = cv2.VideoCapture(0)
protoPath = "D:/Liveness_Detection/face_detector/deploy.prototxt"
modelPath = "D:/Liveness_Detection/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
model = load_model("D:\Liveness_Detection\liveness.model")
le = pickle.loads(open("D:/Liveness_Detection/le.pickle", "rb").read())

@app.route('/')
def index():
    return "Default Message"

def gen(video):
    while True:
        success, image = video.read()
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.8:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")                             
                endX = min(w, endX)
                endY = min(h, endY)
                face = image[startY:endY, startX:endX]
                img_face = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_face = im.fromarray(img_face)
                img_face = img_face.resize((224,224))
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)
                preds = model.predict(face)[0]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                j = np.argmax(preds)
                label = le.classes_[j]
                label = "{}: {:.4f}".format(label, preds[j])
                cv2.putText(image, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(image, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
    