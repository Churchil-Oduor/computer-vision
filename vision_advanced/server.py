import asyncio
import websockets
import cv2
import numpy as np

async def handle_connection(websocket, path):
    print("Client connected")
    try:
        async for message in websocket:
            # Convert binary message to numpy array
            nparr = np.frombuffer(message, np.uint8)
            # Decode image (assuming YUV format from CameraX)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                # Process frame (e.g., convert to grayscale)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Frame', gray)
                cv2.waitKey(1)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

async def main():
    server = await websockets.serve(handle_connection, "0.0.0.0", 8765)
    print("WebSocket server started on ws://0.0.0.0:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
