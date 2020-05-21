import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#looking for the path on the system where the file is
image_dir = os.path.join(BASE_DIR, "images")
#goes to image folder

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_default.xml')
eye_casacade= cv2.CascadeClassifier('Cascades/data/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() #face recognizer
current_id = 0
label_ids ={}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()
			#print(label, path)
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			print(label_ids)
			#y_labels.append(label) # some number
			#x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
			pil_image = Image.open(path).convert("L") # grayscale
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")
			#print(image_array)
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=2, minNeighbors=1)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)
print(y_labels)
print(x_train)
 #PIL is python image library
 #converts the image to grayscale
 # #Converts the image  into HEX
 #  #finding the face  
 #selecting the face regio
with open("pickle/labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f) #saves the id's to pickle

recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizers/trainner.yml")