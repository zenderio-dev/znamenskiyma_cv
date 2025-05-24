import cv2
import numpy as np
import zmq

adress = "84.237.21.36"
port = 6002

comtext = zmq.Context()
socket = comtext.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")
socket.connect(f"tcp://{adress}:{port}")

cv2.namedWindow("Client", cv2.WINDOW_GUI_NORMAL)
count = 0

while True:
    message = socket.recv()
    frame = cv2.imdecode(np.frombuffer(message, np.uint8), -1)
    count += 1

    # LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Маска по яркости
    _, l_mask = cv2.threshold(l, 110, 255, cv2.THRESH_BINARY)

    # Маска по цвету отсекаем серое
    a_dev = cv2.absdiff(a, 128)
    b_dev = cv2.absdiff(b, 128)
    sat_mask = cv2.bitwise_or(a_dev, b_dev)
    _, color_mask = cv2.threshold(sat_mask, 15, 255, cv2.THRESH_BINARY)

    # две маски в одной
    mask = cv2.bitwise_and(l_mask, color_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_count = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 1000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{object_count + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        object_count += 1

    cv2.putText(frame, f"Объекты: {object_count}", (10, 110), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Кадр: {count}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    # Отображение
    cv2.imshow("Client", frame)
    cv2.imshow("Mask", clean)

    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "q":
        break