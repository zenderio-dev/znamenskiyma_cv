import socket
import numpy as np
import matplotlib.pyplot as plt

host = "84.237.21.36"
port = 5152

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

for i in range(10):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))

        sock.send(b"get")

        bts = recvall(sock, 40002)
        if bts is None:
            continue

        rows, cols = bts[0], bts[1]
        img = np.frombuffer(bts[2:rows*cols+2], dtype=np.uint8).reshape(rows, cols)

        pos1_flat_idx = np.argmax(img)
        pos1 = np.unravel_index(pos1_flat_idx, img.shape)

        mask = np.zeros_like(img, dtype=bool)
        center_row, center_col = pos1
        radius = 5
        for r in range(max(0, center_row - radius), min(rows, center_row + radius + 1)):
            for c in range(max(0, center_col - radius), min(cols, center_col + radius + 1)):
                mask[r, c] = True

        img_masked = img.copy()
        img_masked[mask] = 0

        pos2_flat_idx = np.argmax(img_masked)
        pos2 = np.unravel_index(pos2_flat_idx, img_masked.shape)

        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        rounded_distance = np.round(distance, 1)

        result_str = f"{rounded_distance}"
        sock.send(result_str.encode())
        feedback = sock.recv(1024)

        print(f"{rounded_distance}, {feedback.decode()}")

        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.title(f"Distance: {rounded_distance}")
        plt.show(block=False)

plt.show()
