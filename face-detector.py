import cv2

# Load some pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Choose an image to detect faces in
img = cv2.imread('RDJ.jpg')

# Converting to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Creating Reactangle
for (x,y,w,h) in face_coordinates:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

cv2.imshow('Face Detector',img)
cv2.waitKey()
cv2.destroyAllWindows()

print("Success")