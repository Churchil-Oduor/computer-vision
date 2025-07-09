import socket
import cv2
import numpy as np

s = socket.socket()
s.bind(('0.0.0.0', 8888))
s.listen(1)
conn, addr = s.accept()

while True:
    length_bytes = conn.recv(2)
    if not length_bytes:
        break
    length = (length_bytes[0] << 8) + length_bytes[1]
    data = b''
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            break
        data += packet
    if data:
        npimg = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        cv2.imshow("Received Frame", img)
        if cv2.waitKey(1) == ord('q'):
            break

conn.close()
s.close()

