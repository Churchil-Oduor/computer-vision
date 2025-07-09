import socket
import cv2
import numpy as np

HOST = '0.0.0.0'
PORT = 8888

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(5)

print("Listening on port", PORT)
conn, addr = s.accept()

while True:
    length = conn.recv(2)
    if not length:
        break
    size = (length[0] << 8) + length[1]
    data = b''
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            break
        data += packet
    if data:
        npimg = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        cv2.imshow('Live', frame)
        if cv2.waitKey(1) == ord('q'):
            break

conn.close()
s.close()
cv2.destroyAllWindows()

