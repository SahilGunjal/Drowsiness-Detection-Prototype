"""
This code is a Prototype for Drowsiness Detection
Authors: Siddhesh Abhijeet Dhonde (sd1386), Sahil Sanjay Gunjal (sg2736), Atharva Manoj Chiplunkar (ac2434),
        Vaibhav Sharma (vs1654)
"""

import webbrowser
import cv2
import dlib
import imutils
import vlc
from imutils import face_utils
from scipy.spatial import distance


def eye_aspect_ratio(eye):
    """
    This function calculates Eye Aspect Ratio using euclidean distance.
    :param eye: Eye points on the face
    :return:
    """
    return ((distance.euclidean(eye[1], eye[5]) + distance.euclidean(eye[2], eye[4])) / (
            2 * distance.euclidean(eye[0], eye[3])))


def mouth_aspect_ratio(mouth):
    """
    This function calculates Mouth Aspect Ratio using euclidean distance.
    :param mouth: Mouth points on the face
    :return:
    """
    mar = (distance.euclidean(mouth[2], mouth[10]) + distance.euclidean(mouth[4], mouth[8])) / (
            2 * distance.euclidean(mouth[0], mouth[6]))
    print(mar)
    return mar


def extracting_eye_and_face_features():
    """
    This function extract required facial features such as left eye, right eye, mouth points
    :return: returns predictor and all required facial feature points
    """
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    left_eye_start, left_eye_end = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    right_eye_start, right_eye_end = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    mouth_start, mouth_end = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
    return predictor, (left_eye_start, left_eye_end), (right_eye_start, right_eye_end), (mouth_start, mouth_end)


def drowsiness_detection():
    """
    This function performs drowsiness detection.
    :return: None
    """

    # Turning On Camera to capture video.
    captured_frame = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()  # Library for Extracting human face from frame
    predictor, left_eye_points, right_eye_points, mouth_points = extracting_eye_and_face_features()
    drowsy_eye_threshold = 0.2675092419954906  # Best Selected Threshold for Eyes
    yawning_mouth_threshold = 0.5855645444146671  # Best Selected Threshold for Mouth
    continues_frame_threshold = 15
    mouth_frame_threshold = 7
    map_open = 0
    yawn_count = 0
    map_open_flag = 1
    flag = 0
    focus_alert = vlc.MediaPlayer('focus_on_driving.mp3') # playes alert sound
    while True:
        # Press Q to quite the code
        if cv2.waitKey(1) == ord('Q') or cv2.waitKey(1) == ord('q'):
            break

        ret, frame = captured_frame.read()
        frame = imutils.resize(frame, width=1000, height=1000)  # Defining size of frame
        face_found = detector(frame, 0)  # Detects face from frame

        if len(face_found) > 0:
            # Goes in this condition if face found in frame
            face_shape = face_utils.shape_to_np(predictor(frame, face_found[0]))  # Extracting all facial features.
            left_eye = face_shape[left_eye_points[0]:left_eye_points[1]]  # Extracting points for left eye
            right_eye = face_shape[right_eye_points[0]:right_eye_points[1]]  # Extracting points for right eye
            mouth = face_shape[mouth_points[0]:mouth_points[1]]  # Extracting points for Mouth
            leftEyeBorder = cv2.convexHull(left_eye)  # creating convex hull around the left eye points
            rightEyeBorder = cv2.convexHull(right_eye)  # creating convex hull around the roght eye points
            mouthBorder = cv2.convexHull(mouth)  # creating convex hull around the mouth points
            left_EAR = eye_aspect_ratio(left_eye)  # Calculating EAR for left eye
            right_EAR = eye_aspect_ratio(right_eye)  # Calculating EAR for right eye
            average_EAR = (left_EAR + right_EAR) / 2.0  # Taking average of both EAR
            eye_color = (0, 255, 0)
            mouth_color = (0, 255, 0)

            if mouth_aspect_ratio(mouth) > yawning_mouth_threshold:
                # Goes in this condition if Yawn Detected
                cv2.putText(frame, "Yawn Detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                yawn_count = 1
                mouth_color = (255, 0, 0)
                map_open += 1

            elif average_EAR < drowsy_eye_threshold:
                # Goes in this condition if drowsiness Detected
                flag += 1
                eye_color = (0, 255, 255)

                if map_open_flag:  # Map flag
                    map_open_flag = 0
                    map_open += 1  # Map counter

                if yawn_count and flag >= mouth_frame_threshold:
                    # Goes in this condition if drowsiness Detected with Yawning
                    eye_color = (255, 0, 0)
                    mouth_color = (255, 0, 0)
                    cv2.putText(frame, "Drowsy after yawn", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                    focus_alert.play()
                    map_open += 1

                elif flag >= continues_frame_threshold:
                    # Goes in this condition if drowsiness Detected for certain number of Frames. This throws an alert
                    eye_color = (255, 0, 0)
                    cv2.putText(frame, "Drowsiness Detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                    focus_alert.play()
                    map_open += 1

            elif average_EAR > drowsy_eye_threshold and flag:
                # Condition to reset all flags and counters. Goes in this if person is not drowsy
                flag = 0
                eye_color = (0, 255, 0)
                yawn_count = 0
                focus_alert.stop()
                map_open_flag = 1

            if average_EAR > drowsy_eye_threshold:
                # Condition to stop alert
                focus_alert.stop()

            cv2.drawContours(frame, [leftEyeBorder], -1, eye_color, 2)
            cv2.drawContours(frame, [rightEyeBorder], -1, eye_color, 2)
            cv2.drawContours(frame, [mouthBorder], -1, mouth_color, 2)

            if map_open > 100:
                # This condition open Google Maps showing nearby Hotels or Motels.
                map_open = 0
                map_open_flag = 1
                webbrowser.open("https://www.google.com/maps/search/hotels+or+motels+near+me")
        cv2.imshow('Drivers Drowsiness', frame)

    captured_frame.release()
    cv2.destroyAllWindows()


def main():
    """
    This is main function
    :return:
    """
    drowsiness_detection()


if __name__ == '__main__':
    main()
