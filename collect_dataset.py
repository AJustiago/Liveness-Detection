import numpy as np
import cv2
import os

print("[INFO] loading face detector...")
protoPath = "D:/Liveness_Detection/face_detector/deploy.prototxt"
modelPath = "D:/Liveness_Detection/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

vs = cv2.VideoCapture("D:/Liveness_Detection/videos/fake.mp4")

read = 0
saved = 0
while True:
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	read += 1
	if read % 1 != 0: # read % num of frame to skip
		continue
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	if len(detections) > 0:
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]
			p = os.path.sep.join(['dataset/fake',
				"{}.png".format(saved)])
			cv2.imwrite(p, face)
			saved += 1
			print("[INFO] saved {} to disk".format(p))

vs.release()
cv2.destroyAllWindows()