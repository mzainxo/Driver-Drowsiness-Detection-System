import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import pygame
from threading import Thread
import time
from datetime import datetime

classes = ['Closed', 'Open']
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")
cap = cv2.VideoCapture(0)
model = load_model("ds_project.h5")
count = 0
alarm_on = False
alarm_sound = "data/wake_up_alarm.mp3"
msg_sound = "data/msg.mp3"
wel_sound="data/welcome.mp3"
status1 = ''
status2 = ''
welcome_played = False

def start_alarm(sound, play_on_loop=True):
    """Play the alarm sound"""
    pygame.mixer.init()
    pygame.mixer.music.load(sound)
    if play_on_loop:
        pygame.mixer.music.play(-1)  # Play on loop
    else:
        pygame.mixer.music.play()

def stop_alarm():
    """Stop the alarm sound"""
    pygame.mixer.music.stop()
    time.sleep(2)
    play_msg_once(msg_sound,"It is advisable to consider taking a break or resting.")

def play_msg_once(sound,msg):
    """Play the message sound once"""
    pygame.mixer.init()
    pygame.mixer.music.load(sound)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        cv2.putText(frame, f"{msg}", (7, height-30), cv2.FONT_ITALIC, 0.4, (255, 255, 255), 1)
        continue


while True:
    _, frame = cap.read()
    height, width, _ = frame.shape

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw background for the text
    cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"{current_time}", (400, 20), cv2.FONT_ITALIC, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Drowsiness Detection System", (10, 20), cv2.FONT_ITALIC, 0.6, (255, 255, 255), 1)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)

        if not welcome_played:
            time.sleep(1)
            txt = "Hi! I'm your vigilant companion on the road and I'll be watching out for you while you are driving."
            t = Thread(target=play_msg_once, args=(wel_sound, txt))
            t.daemon = True
            welcome_played = True   # Set the flag to True after playing the welcome sound
            t.start()
             

        for (x1, y1, w1, h1) in left_eye:
            cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
            eye1 = roi_color[y1:y1+h1, x1:x1+w1]
            eye1 = cv2.resize(eye1, (145, 145))
            eye1 = eye1.astype('float') / 255.0
            eye1 = img_to_array(eye1)
            eye1 = np.expand_dims(eye1, axis=0)
            pred1 = model.predict(eye1)
            status1=np.argmax(pred1)
            break

        for (x2, y2, w2, h2) in right_eye:
            cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
            eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
            eye2 = cv2.resize(eye2, (145, 145))
            eye2 = eye2.astype('float') / 255.0
            eye2 = img_to_array(eye2)
            eye2 = np.expand_dims(eye2, axis=0)
            pred2 = model.predict(eye2)
            status2=np.argmax(pred2)
            break

        # If the eyes are closed, start counting
        if status1 == 2 and status2 == 2:
            count += 1
            cv2.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 48), cv2.FONT_ITALIC, 0.9, (0, 0, 255), 2)
            # if eyes are closed for 4 consecutive frames, start the alarm
            if count >= 4:
                cv2.putText(frame, "Warning: Drowsiness Detected!", (80, height-20), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
                if not alarm_on:
                    alarm_on = True
                    # play the alarm sound in a new thread
                    t = Thread(target=start_alarm, args=(alarm_sound, True))
                    t.daemon = True
                    t.start()
        else:
            cv2.putText(frame, "Eyes Open", (10, 48), cv2.FONT_ITALIC, 0.9, (0, 255, 0), 2)
            count = 0
            if alarm_on:
                alarm_on = False
                
                # stop the alarm on a new thread
                t = Thread(target=stop_alarm)
                t.daemon = True
                t.start()
    cv2.imshow("Data Science Project by Zain, Hamza, & Askari - BSE-5(B)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#stop_alarm()
cap.release()
cv2.destroyAllWindows()