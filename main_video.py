import cv2
from simple_facerec import SimpleFacerec

# Encode faces from the "images/" folder
sf = SimpleFacerec()
sf.load_encoding_images("images/")

# Initialize camera capture with specified resolution
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set camera width
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set camera height

while True:
    # Read a frame from the camera
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)  # Flip frame for mirror effect

    # Detect known faces in the frame
    face_locations, face_names = sf.detect_known_faces(frame)

    # Loop through detected faces and their names
    for face_loc, name in zip(face_locations, face_names):
        # Extract face coordinates
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Display the name of the recognized face on the frame
        cv2.putText(frame, name, (x1, y1 - 8), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)

    # Show the frame with detected faces
    cv2.imshow("Frame", frame)

    # Exit the loop when 'Esc' key is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release camera resources and close all OpenCV windows
vid.release()
cv2.destroyAllWindows()
