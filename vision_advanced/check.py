import socket
import cv2
import numpy as np
import struct

HOST = '0.0.0.0'  # Accept from any IP
PORT = 8888

def recv_exact(sock, length):
    """Receive exactly 'length' bytes from socket."""
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            return None
        data += more
    return data

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"Listening on {HOST}:{PORT}...")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    with conn:
        while True:
            # Receive the 4-byte length header
            length_bytes = recv_exact(conn, 4)
            if not length_bytes:
                break
            frame_length = struct.unpack('>I', length_bytes)[0]

            # Receive the actual JPEG frame
            frame_data = recv_exact(conn, frame_length)
            if not frame_data:
                break

            # Decode the image
            np_data = np.frombuffer(frame_data, dtype=np.uint8)
            img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            if img is not None:
                cv2.imshow("Android Stream", img)
                if cv2.waitKey(1) == 27:  # ESC to quit
                    break

    cv2.destroyAllWindows()
    server_socket.close()

if __name__ == "__main__":
    start_server()

