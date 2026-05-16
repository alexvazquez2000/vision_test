#This is the fully working code
# The video to accompany this GIST is at: https://youtu.be/41przp5Dm7M


#These need to be pip installed.
import cv2
import mediapipe as mp

grid = True

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize MediaPipe drawing utilities for visualization
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to draw a grid with coordinate labels on the frame
def draw_grid_with_labels(frame, num_rows=10, num_cols=10):
    h, w = frame.shape[:2]
    # Draw horizontal lines and y-coordinates
    for i in range(1, num_rows):
        y = h * i // num_rows
        cv2.line(frame, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(frame, str(y), (10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 1)
    # Draw vertical lines and x-coordinates
    for i in range(1, num_cols):
        x = w * i // num_cols
        cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(frame, str(x), (x + 5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 1)

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the image horizontally for later selfie-view display
    frame = cv2.flip(frame, 1)

    # Draw grid with coordinate labels on the frame
    if cv2.waitKey(1) == ord('g'): grid = True
    if cv2.waitKey(1) == ord('h'): grid = False
    if grid: draw_grid_with_labels(frame)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)

    # Draw hand landmarks and print thumb coordinates
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw all hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

            # Get thumb landmarks (landmark indices 1 to 4)
            thumb_landmarks = [hand_landmarks.landmark[i] for i in range(1, 5)]

            # Thumb tip is landmark 4 and thumb base is landmark 2
            thumb_tip = thumb_landmarks[3]
            thumb_base = thumb_landmarks[1]

            # Convert landmark position to pixel coordinates
            thumb_tip_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            thumb_base_coords = (int(thumb_base.x * frame.shape[1]), int(thumb_base.y * frame.shape[0]))

            # Check if the thumb is up or down
            if thumb_tip_coords[1] < thumb_base_coords[1]:
                print("Thumb is up!")
                cv2.putText(frame, "Thumb Up!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                print("Thumb is down!")
                cv2.putText(frame, "Thumb Down!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition with Grid and Coordinates', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
