import cv2

# Load some pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Capturing the video
vid = cv2.VideoCapture(0)


while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    
    #converting frame to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)


    for (x,y,w,h) in face_coordinates:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()

print("Success")