import asyncio
import websockets
import os

async def receive_weights(websocket, path):
    # Generate a unique filename for each connection
    client_address = websocket.remote_address[0]
    filename = f"weights_{client_address.replace('.', '_')}.pth"
    filepath = os.path.join('./received', filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
   
    # Open a file to write the incoming chunks
    with open(filepath, 'wb') as f:
        async for message in websocket:
            if message == "EOF":
                print(f"Finished receiving file from {client_address}")
                break
            f.write(message)
            print(f"Received a chunk from {client_address}")

    # Notify when a file is completely received
    print(f"Received complete file from {client_address} saved as {filename}")

async def main():
    # Create the server, specifying the handler function and IP address to bind
    async with websockets.serve(receive_weights, "192.168.8.160", 8765):
        print("Server started, listening on 192.168.8.160:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())