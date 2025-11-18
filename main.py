import cv2
from fer.fer import FER

def main():
    # Start webcam
    cam = cv2.VideoCapture(0)

    # Initialize emotion detector
    detector = FER(mtcnn=True)

    print("Press 'q' to quit")

    while True:
        # Capture a frame from webcam
        ret, frame = cam.read()
        if not ret:
            break

        # Detect emotions in the frame
        emotions = detector.detect_emotions(frame)

        # Draw box and label for each detected face
        for face in emotions:
            (x, y, w, h) = face["box"]
            # Find emotion with highest score
            dominant_emotion = max(face["emotions"], key=face["emotions"].get)
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Write the emotion label
            cv2.putText(frame, dominant_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the live video
        cv2.imshow("Emotion Detector", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
