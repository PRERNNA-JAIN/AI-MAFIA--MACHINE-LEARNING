# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
face_data = []
dataset_path = "C:\Users\cheta\Downloads"
file_name = input("enter the name of the person: ")
while True:
    ret, frame = cap.read()
    if ret == False:
        continue
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        continue
    faces = sorted(faces, key=lambda f: f[2] * f[3])
    # pick the last face bcoz it has large area
    for face in faces[-1:]:
        # drawing the bound box or the rectangle
        x, y, w, h = face
        cv2.rectangle(gray_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # extract (crop out the required face):
        offset = 10
        face_section = gray_frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))
        face_data.append(face_section)
        print(len(face_section))
    # cv2.imshow("frame",frame)
    cv2.imshow("gray_frame", gray_frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
# convert face data list into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# save this data into file system
np.save(dataset_path + file_name + '.npy', face_data)
print("data saved successfully!")
cap.release()
cap.destroyAllWindows()
