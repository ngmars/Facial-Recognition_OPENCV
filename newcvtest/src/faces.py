import numpy as np 
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_default.xml')
eye_casacade= cv2.CascadeClassifier('Cascades/data/haarcascade_eye.xml')
pface_cascade=cv2.CascadeClassifier('Cascades/data/haarcascade_lowerbody.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() #face recognizer
recognizer.read("recognizers/trainner.yml")
labels ={}
with open("pickle/labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels={v:k for k,v in og_labels.items()}
cap = cv2.VideoCapture(0)
while(True):
	#display frame by frame
	ret,frame = cap.read()
	#Can only be implemented in gray--casacade only accepts that, assumption made by trial and error
	gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, minNeighbors=5)
	for (x, y, w, h) in faces:
		#print(x,y,w,h)
		end_cord_x= x + w
		end_cord_y= y + h
		#uses the x(length) and y(height) h(y coordinate end) w( X coordinate end)coordinates 
		#to select the face and isolate only that region
		roi_gray = gray[y:end_cord_y, x:end_cord_x] 
		roi_color = gray[y:y+h, x:x+w]
		id_,confidence= recognizer.predict(roi_gray)# conf = confidence in model
		if confidence>=10:
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color= (255,255,255)
			stroke =2
			cv2.putText(frame,name,(x,y),font, 1, color, stroke, cv2.LINE_AA)
		img_item= "my-imagei.png"
		cv2.imwrite(img_item,roi_color)#creates the image
		color = (255,0,0) #Blue,Green,Red
		stroke = 2 #thickness of line
		
		cv2.rectangle(frame,(x, y), (end_cord_x, end_cord_y),color, stroke )
		eyes = eye_casacade.detectMultiScale(roi_gray, scaleFactor= 1.5,minNeighbors=10,minSize=(5, 5),)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		
		

	#Display the resulting frame
	cv2.imshow('frame',frame)
	if  cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

