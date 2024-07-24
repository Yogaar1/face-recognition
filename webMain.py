# Import library
import streamlit.components.v1 as components
import streamlit as st
import face_recognition
from datetime import datetime
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
import time

# Setup layout website Streamlit
FRAME_WINDOW = st.image([])

# Hide Streamlit menu
hide_st_style = """ 
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

menu = ["HOME", "LOGIN", "REGISTER", "DATA"]  # Menu
choice = st.sidebar.selectbox("Menu", menu)  # Sidebar menu

# Set directory for storing face data
path = 'data'
images = []
classNames = []
myList = os.listdir(path) if os.path.exists(path) else []

# Login Page
col1, col2, col3 = st.columns(3)  # Columns
cap = cv2.VideoCapture(0)  # Capture video

if choice == 'LOGIN': 
    st.markdown("<h2 style='text-align: center; color: black;'>ATTENDANCE</h2>", unsafe_allow_html=True)  # Title
    with col1:  # Column 1
        st.subheader("LOGIN")
        run = st.checkbox("Run camera")  # Checkbox

    if run:
        # Load images from directory
        for cl in myList:
            curlImg = cv2.imread(f'{path}/{cl}')
            images.append(curlImg)
            classNames.append(os.path.splitext(cl)[0])  # Extract name from filename
        print(classNames)

        # Function to find face encodings
        def find_encodings(images):
            encode_list = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                encode_list.append(encode)
            return encode_list

        # Function to update attendance list
        def face_list(name):
            with open('absensi.csv', 'a+') as f:
                f.seek(0)
                my_data_list = f.readlines()
                name_list = [line.split(',')[0] for line in my_data_list]
                if name not in name_list:
                    now = datetime.now()
                    dt_string = now.strftime('%H:%M:%S')
                    f.writelines(f'\n{name},{dt_string}')

        encode_list_known = find_encodings(images)
        print('Encoding complete!')

        # Capture frames from camera and detect faces
        while True:
            success, img = cap.read()
            if not success:
                st.error("Failed to capture image from the camera.")
                break
            
            img_s = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
            face_cur_frame = face_recognition.face_locations(img_s)
            encode_cur_frame = face_recognition.face_encodings(img_s, face_cur_frame)

            for encode_face, face_loc in zip(encode_cur_frame, face_cur_frame):
                matches = face_recognition.compare_faces(encode_list_known, encode_face)
                face_dis = face_recognition.face_distance(encode_list_known, encode_face)
                match_index = np.argmin(face_dis)

                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                if matches[match_index]:
                    name = classNames[match_index].upper()
                    print(name)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    face_list(name)
                    time.sleep(3)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            FRAME_WINDOW.image(img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        pass

# Register menu
elif choice == 'REGISTER':
    with col2:
        st.subheader("REGISTER")

    def load_image(image_file):
        img = Image.open(image_file)
        return img

    image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
    if image_file is not None:
        file_details = {"FileName": image_file.name, "FileType": image_file.type}
        st.write(file_details)
        img = load_image(image_file)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, image_file.name), "wb") as f: 
            f.write(image_file.getbuffer())         
        st.success("Saved File")

# Read data menu
elif choice == 'DATA':
    with col2:
        if os.path.exists('absensi.csv'):
            df = pd.read_csv('absensi.csv')
            st.subheader("READ DATA")
            st.write(df)
        else:
            st.warning("No attendance data found.")

elif choice == 'HOME':
    st.subheader("Machine Learning Face Recognition with HOG Method")
    st.image("fc.png", width=700) 
    st.markdown("""
    The HOG (Histogram of Oriented Gradients) method is a technique frequently used in image processing and computer vision to detect and recognize objects in images. HOG has proven to be highly effective in various applications, especially in object detection and pattern recognition.

    This method converts the image into a gradient-based histogram representation that can be used to train object detection models, such as face detectors or pedestrian detectors. Here are some reasons why HOG is used for face recognition:
    - HOG captures gradient information that is invariant to changes in illumination.
    - HOG focuses on local gradients in the image, allowing the detection of important facial features, such as nose edges, mouth, eyes, and face contours.
    - This method is robust to minor variations in rotation and scale.
    - HOG can be applied quickly and used in real-time applications, such as face detection in surveillance cameras.
    - HOG is easy to implement and understand, both for researchers and practitioners.

    In summary, HOG offers many advantages and benefits for face recognition applications.

    Yoga Ari Nugroho 21.11.4128
    """)

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
