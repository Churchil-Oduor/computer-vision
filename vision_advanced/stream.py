import socket
import cv2
import numpy as np

server_socket = socket.socket()
server_socket.bind(("0.0.0.0", 8888))
server_socket.listen(1)

print("Waiting for connection...")
conn, addr = server_socket.accept()
print(f"Connected by {addr}")

while True:
    # Read 2 bytes for image length
    length_bytes = conn.recv(2)
    if not length_bytes:
        break
    length = (length_bytes[0] << 8) + length_bytes[1]

    # Now receive the actual image
    data = b''
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            break
        data += packet

    # Decode JPEG
    np_arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is not None:
        cv2.imshow('Received Frame', frame)
        if cv2.waitKey(1) == 27:  # Press ESC to exit
            break

conn.close()
server_socket.close()

