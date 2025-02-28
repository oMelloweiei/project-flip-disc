import asyncio
import websockets
import json
import random 

# WebSocket server that sends a 36x24 bitmatrix
# Updated function without the 'path' parameter
async def send_bitmatrix(websocket):
    connection_id = id(websocket)
    print(f"Client connected: {connection_id}")
    try:
        while True:
            # Example: generate a random 36x24 bitmatrix (0 or 1)
            bitmatrix = [
                [random.randint(0, 1) for _ in range(36)] for _ in range(24)
            ]
            # Send the bitmatrix as a JSON object
            await websocket.send(json.dumps(bitmatrix))
            print(f"Matrix sent to client {connection_id}")
            await asyncio.sleep(1)  # Send every 1 second
    except websockets.ConnectionClosed:
        print(f"Client disconnected: {connection_id}")

# Start the WebSocket server on ws://localhost:8765
async def main():
    async with websockets.serve(send_bitmatrix, "localhost", 8765):
        print("WebSocket server started at ws://localhost:8765")
        await asyncio.Future()  # Run forever

# Run the server
if __name__ == "__main__":
    asyncio.run(main())
