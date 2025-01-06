import cv2
import os

# Create the dataset directory if it doesn't exist
dataset_dir = 'dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Initialize the webcam
video = cv2.VideoCapture(0)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize ID for the dataset
id = input("Enter Your ID: ")

# Count the number of captured images
count = 0

while True:
    # Capture frame-by-frame
    ret, frame = video.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces and save them to the dataset
    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]  # Extract the face region
        cv2.imwrite(f'dataset/User.{id}.{count}.jpg', face_img)  # Save the face image
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed or 10 images are captured
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 10:
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()

print(".......................Dataset Collection Done..................")
