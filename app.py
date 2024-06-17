import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe pose and selfie segmentation
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
pose = mp_pose.Pose()
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture("video.mp4")

def draw_3d_landmarks_on_box(image, landmarks, box_position):
    h, w, _ = image.shape
    box_w, box_h = 200, 200  # Size of the box
    box_x, box_y = box_position
    
    # the box
    box_image = np.ones((box_h, box_w, 3), dtype=np.uint8) * np.array([255, 255, 255], dtype=np.uint8)
    
    for landmark in landmarks:
        x = int(landmark.x * box_w)
        y = int(landmark.y * box_h)
        z = int(landmark.z * box_w)  # assuming box_w == box_h for aspect ratio consistency
        cv2.circle(box_image, (x, y), 3, (255, 0, 0), -1)
        cv2.putText(box_image, f'  {z:.2f}', (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (51, 51, 255), 1, cv2.LINE_AA)

    # Overlay the box image on the main image
    image[box_y:box_y+box_h, box_x:box_x+box_w] = box_image



def draw_3d_landmarks_coords_on_box(image, landmarks, box_position):
    h, w, _ = image.shape
    box_w, box_h = 200, 200  # Size of the box
    box_x, box_y = box_position
    
    # box_image
    box_image = np.ones((box_h, box_w, 3), dtype=np.uint8) * np.array([255, 255, 255], dtype=np.uint8)
    
    for landmark in landmarks:
        x = int(landmark.x * box_w)
        y = int(landmark.y * box_h)
        z = int(landmark.z * box_w)  # assuming box_w == box_h for aspect ratio consistency
        cv2.putText(box_image, f'  {z:.2f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

    # Overlay the box image on the main image
    image[box_y:box_y+box_h, box_x:box_x+box_w] = box_image

def main():
    frame_count = 0

    # Create a resizable window
    cv2.namedWindow('Motion Capture', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Restart the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect the pose
        results_pose = pose.process(image_rgb)
        results_segmentation = selfie_segmentation.process(image_rgb)

        if results_pose.pose_landmarks:
            # Draw the pose annotation on the image.
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract landmarks
            landmarks = results_pose.pose_landmarks.landmark

            # Draw 3D landmarks in the box
            draw_3d_landmarks_on_box(frame, landmarks, (frame.shape[1] - 210, 10))  # 10 px padding

            # draw_3d_landmarks_coords_on_box(frame, landmarks, (frame.shape[1] - 460, 10))

            # Perform analysis (posture analysis, gait detection)
            # Calculate angle between shoulder, elbow, and wrist
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

        # Apply grayscale to the background
        mask = results_segmentation.segmentation_mask > 0.1
        mask = np.stack((mask,) * 3, axis=-1)  # Convert to 3 channels
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert grayscale back to BGR
        frame = np.where(mask, frame, background)

        # Display the resulting frame
        cv2.imshow('Motion Capture', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
