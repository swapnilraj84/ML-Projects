import cv2  # OpenCV library for computer vision tasks
import mediapipe as mp  # Mediapipe library for hand detection and tracking
import pyautogui  # PyAutoGUI library for simulating keyboard and mouse actions
from scipy.odr import Output  # Optional import for data fitting (not used in this code)

# Initialize variables to store coordinates of specific landmarks
x1 = y1 = x2 = y2 = 0

# Start capturing video from the webcam
webcam = cv2.VideoCapture(0)

# Initialize Mediapipe Hands module for hand detection and tracking
my_hands = mp.solutions.hands.Hands()

# Utility functions for drawing landmarks on the hand
drawing_utils = mp.solutions.drawing_utils

while True:  # Infinite loop for real-time processing
    _, image = webcam.read()  # Read a frame from the webcam
    image = cv2.flip(image, 1)  # Flip the frame horizontally for a mirror view

    # Get the dimensions of the frame
    frame_height, frame_width, _ = image.shape

    # Convert the image from BGR to RGB format for Mediapipe processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the RGB image to detect hands and their landmarks
    output = my_hands.process(rgb_image)

    # Get the detected hand landmarks from the output
    hands = output.multi_hand_landmarks

    if hands:  # If hands are detected
        for hand in hands:  # Loop through each detected hand
            # Draw landmarks and connections on the hand
            drawing_utils.draw_landmarks(image, hand)

            # Extract the list of landmarks for the current hand
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):  # Loop through each landmark
                # Convert normalized landmark coordinates to pixel coordinates
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # If the landmark is the tip of the index finger
                    # Draw a yellow circle at the tip of the index finger
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 255, 255), thickness=3)
                    x1 = x  # Store x-coordinate of the index finger tip
                    y1 = y  # Store y-coordinate of the index finger tip

                if id == 4:  # If the landmark is the tip of the thumb
                    # Draw a red circle at the tip of the thumb
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 0, 255), thickness=3)
                    x2 = x  # Store x-coordinate of the thumb tip
                    y2 = y  # Store y-coordinate of the thumb tip

        # Calculate the Euclidean distance between the index finger tip and thumb tip
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 // 4

        # Draw a green line between the index finger tip and thumb tip
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Adjust system volume based on the distance between the fingertips
        if dist > 50:  # If the distance is greater than 50 (fingers apart)
            pyautogui.press("volumeup")  # Simulate pressing the "volume up" key
        else:  # If the distance is less than or equal to 50 (fingers close)
            pyautogui.press("volumedown")  # Simulate pressing the "volume down" key

    # Display the processed video frame with annotations
    cv2.imshow("Hand Volume Control using Python", image)

    key = cv2.waitKey(10)  # Wait for 10ms for a key press
    if key == 27:  # If the 'Esc' key is pressed
        break  # Exit the loop

# Release the webcam resource
webcam.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
