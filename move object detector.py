import cv2   # OpenCV library
import time  # for delay
from imutils import resize  # for resizing frames

# Initialize camera (0 = default webcam, 1 = external webcam)
cam = cv2.VideoCapture(0)
time.sleep(1)

firstFrame = None
area = 500   # minimum area size for motion detection

while True:
    # Read frame from camera
    ret, img = cam.read()
    if not ret:
        break

    text = "Normal"
    img = resize(img, width=500)   # resize for consistency

    # Convert to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur (to reduce noise and improve detection)
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)

    # Capture first frame (reference background)
    if firstFrame is None:
        firstFrame = gaussianImg
        continue

    # Calculate absolute difference between first frame and current frame
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)

    # Apply threshold
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilation to fill in gaps
    threshImg = cv2.dilate(threshImg, None, iterations=2)

    # Find contours (areas of motion)
    contours, _ = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object Detected"

    # Show text on frame
    cv2.putText(img, text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show video windows
    cv2.imshow("Camera Feed", img)
    cv2.imshow("Threshold", threshImg)
    cv2.imshow("Frame Difference", imgDiff)

    key = cv2.waitKey(10)
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
