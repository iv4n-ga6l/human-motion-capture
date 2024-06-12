# pip install opencv-python mediapipe numpy matplotlib


import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture("video.mp4")

def plot_3d_landmarks(landmarks):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xs = [landmark.x for landmark in landmarks]
    ys = [landmark.y for landmark in landmarks]
    zs = [landmark.z for landmark in landmarks]
    
    ax.scatter(xs, ys, zs)
    plt.show(block=False)
    plt.pause(0.001)
    plt.close(fig)

def main():
    frame_count = 0
    display_frequency = 60  # Change this value to control how often 3D plots are displayed

    while True:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Restart the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                break

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and detect the pose
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # Draw the pose annotation on the image.
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Extract landmarks
                landmarks = results.pose_landmarks.landmark
                
                # Display 3D plot at specified intervals
                # if frame_count % display_frequency == 0:
                #     plot_3d_landmarks(landmarks)

                # Perform analysis here (e.g., posture analysis, gait detection)
                # Example: Calculate angle between shoulder, elbow, and wrist
                shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z])

                elbow = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z])

                wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z])

                angle = np.arccos(np.dot(elbow-shoulder, wrist-elbow) /
                                (np.linalg.norm(elbow-shoulder) * np.linalg.norm(wrist-elbow)))

                angle_deg = np.degrees(angle)
                cv2.putText(frame, f'Elbow Angle: {int(angle_deg)}', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('3D Motion Capture', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

            frame_count += 1

if __name__ == "__main__":
    main()
