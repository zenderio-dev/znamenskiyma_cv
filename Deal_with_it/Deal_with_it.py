import cv2
import numpy as np

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, -2)

glass = cv2.imread('deal_with_it.png', cv2.IMREAD_UNCHANGED)


def censore(image, size=(5,5)):
    result = np.zeros_like(image)
    stepy = result.shape[0] // size[0]
    stepx = result.shape[1] // size[1]
    for y in range(0, image.shape[0], stepy):
        for x in range(0, image.shape[1], stepx):
            for c in range(0, image.shape[2]):
                result[y:y + stepy, x:x + stepx, c] = np.mean(image[y:y + stepy, x:x + stepx, c])
    return result

face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade-eye.xml')

while capture.isOpened():
    ret, frame = capture.read()
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=2.3, minNeighbors=12)
    

    if len(eyes) == 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        x1, y1, w1, h1 = eyes[0]
        x2, y2, w2, h2 = eyes[1]
        x = x1
        y = min(y1, y2)
        w = (x2 + w2) - x1
        h = max(y1 + h1, y2 + h2) - y
        
        resized_glasses = cv2.resize(glass, (w, h))
        
        if resized_glasses.shape[2] == 4:
            resized_glasses = resized_glasses[:, :, :3]
        if y >= 0 and y + h <= frame.shape[0] and x + w <= frame.shape[1]:
            frame[y:y+h, x:x+w] = resized_glasses

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        new_w = int(1.5*w)
        new_h = int(1.5*h)
        x -= w//4
        y -= h//4
        try:
            roi = frame[y:y + new_h, x:x + new_w]
            censored = censore(roi, (5, 5))
            frame[y:y + new_h, x:x + new_w] = censored
        except ValueError:
            pass
    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "q":
        break

    cv2.imshow('Camera', frame)

capture.release()
cv2.destroyAllWindows()
