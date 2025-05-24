import cv2
import numpy as np
import random

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
capture = cv2.VideoCapture(0)

color_ranges = {
    "1": { "lower": np.array([0, 150, 100]), "upper": np.array([10, 255, 255]) },    # Красный
    "2": { "lower": np.array([50, 100, 100]), "upper": np.array([70, 255, 255]) },   # Зелёный
    "3": { "lower": np.array([100, 150, 50]), "upper": np.array([130, 255, 255]) }   # Голубой
}

game_started = False
guess_colors = []
guessed = False

def get_ball(hsv_image, lower, upper):
    mask = cv2.inRange(hsv_image, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > 10:
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)
            return True, (int(x), int(y), int(radius), mask, (x_rect, y_rect, w_rect, h_rect))
    return False, (-1, -1, -1, np.array([]), (0, 0, 0, 0))

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    detected_balls = []

    for key, ranges in color_ranges.items():
        found, (x, y, radius, mask, rect) = get_ball(hsv, ranges["lower"], ranges["upper"])
        if found:
        
            cv2.circle(frame, (x, y), radius, (255, 0, 255), 2)
            detected_balls.append((x, key))

    if len(color_ranges) == 3:
        if not game_started:
            guess_colors = list(color_ranges.keys())
            random.shuffle(guess_colors)
            print("Загаданная последовательность", guess_colors)
            game_started = True
        else:
            if len(detected_balls) == 3:
                detected_balls.sort(key=lambda b: b[0]) 
                user_order = [key for _, key in detected_balls]
                guessed = (user_order == guess_colors)
            else:
                guessed = False

    height = frame.shape[0]
    cv2.putText(frame, "Instructions:", (10, height - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, "Show 3 balls in the correct order", (10, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, "Press Q to quit", (10, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    if game_started:
        cv2.putText(frame, "Game started!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    if guessed:
        cv2.putText(frame, "NICE! You are win", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
