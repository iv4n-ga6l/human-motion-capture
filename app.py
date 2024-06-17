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
        cv2.putText(box_image, f'  {z:.2f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (51, 51, 255), 1, cv2.LINE_AA)

    # Overlay the box image on the main image
    image[box_y:box_y+box_h, box_x:box_x+box_w] = box_image


# def draw_3d_landmarks_coords_on_box(image, landmarks, box_position):
#     h, w, _ = image.shape
#     box_w, box_h = 200, 200  # Size of the box
#     box_x, box_y = box_position
    
#     # box_image
#     box_image = np.ones((box_h, box_w, 3), dtype=np.uint8) * np.array([255, 255, 255], dtype=np.uint8)
    
#     for landmark in landmarks:
#         x = int(landmark.x * box_w)
#         y = int(landmark.y * box_h)
#         z = int(landmark.z * box_w)  # assuming box_w == box_h for aspect ratio consistency
#         cv2.putText(box_image, f'  {z:.2f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

#     # Overlay the box image on the main image
#     image[box_y:box_y+box_h, box_x:box_x+box_w] = box_image


def draw_analytics_overlay(landmarks, overlay_size=(300, 200)):
    overlay_image = np.ones((overlay_size[1], overlay_size[0], 3), dtype=np.uint8) * np.array([255, 255, 255], dtype=np.uint8)
    text_y = 20

    ## Posture Analysis
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
    cv2.putText(overlay_image, f'Elbow Angle: {int(angle_deg)}', (10, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    text_y += 30

    ## Gait Analysis
    # Calculate stride length
    left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
    right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
    stride_length = np.linalg.norm([left_heel.x - right_heel.x, left_heel.y - right_heel.y, left_heel.z - right_heel.z])
    cv2.putText(overlay_image, f'Stride Length: {stride_length:.2f}', (10, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    text_y += 30

    ## Balance Analysis
    # Calculate center of mass
    com_x = sum([lmk.x for lmk in landmarks]) / len(landmarks)
    com_y = sum([lmk.y for lmk in landmarks]) / len(landmarks)
    com_z = sum([lmk.z for lmk in landmarks]) / len(landmarks)
    com = np.array([com_x, com_y, com_z])

    # Calculate distance from center of mass to left and right ankles
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    left_ankle_dist = np.linalg.norm([left_ankle.x - com_x, left_ankle.y - com_y, left_ankle.z - com_z])
    right_ankle_dist = np.linalg.norm([right_ankle.x - com_x, right_ankle.y - com_y, right_ankle.z - com_z])

    # Calculate balance score
    balance_score = abs(left_ankle_dist - right_ankle_dist)
    cv2.putText(overlay_image, f'Balance Score: {balance_score:.2f}', (10, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return overlay_image

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

            # Draw analytics overlay
            analytics_overlay = draw_analytics_overlay(landmarks)
            frame[10:10+analytics_overlay.shape[0], 10:10+analytics_overlay.shape[1]] = analytics_overlay

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