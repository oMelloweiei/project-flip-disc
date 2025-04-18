import React, { useEffect, useState } from "react";
import Flipdot from "../components/Flipdot";
import { io } from "socket.io-client";

function Test() {
  const [bitMatrix, setBitMatrix] = useState(
    Array.from({ length: 24 }, () => Array.from({ length: 36 }, () => 0))
  );
  const [isConnected, setIsConnected] = useState(false);
  const [fps, setFps] = useState(0);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [socket, setSocket] = useState(null);

  // Initialize Socket.IO connection
  useEffect(() => {
    // Connect to the Flask-SocketIO server
    const newSocket = io("http://localhost:5000");
    
    // Handle connection events
    newSocket.on("connect", () => {
      console.log("Connected to server");
      setIsConnected(true);
    });
    
    newSocket.on("disconnect", () => {
      console.log("Disconnected from server");
      setIsConnected(false);
    });
    
    // Handle incoming flipdisc data
    newSocket.on("flipdisc_update", (data) => {
      setBitMatrix(data.matrix);
      setFps(data.fps);
      setLastUpdate(new Date().toLocaleTimeString());
    });
    
    setSocket(newSocket);
    
    // Clean up on unmount
    return () => {
      if (newSocket) {
        newSocket.disconnect();
      }
    };
  }, []);

  return (
    <div className="flex flex-col items-center p-4">
      <h1 className="text-2xl font-bold mb-4">U2NET Human Segmentation</h1>
      
      {/* Connection status */}
      <div className="mb-4 flex items-center">
        <span className={`inline-block w-3 h-3 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></span>
        <span>{isConnected ? 'Connected to segmentation server' : 'Disconnected'}</span>
      </div>
      
      {/* FPS and last update time */}
      <div className="mb-4 flex space-x-4">
        {fps > 0 && (
          <div className="text-sm text-gray-600">
            FPS: <span className="font-bold">{fps}</span>
          </div>
        )}
        {lastUpdate && (
          <div className="text-sm text-gray-600">
            Last updated: {lastUpdate}
          </div>
        )}
      </div>
      
      {/* Flipdot display */}
      <div className="border-2 border-gray-300 p-4 rounded">
        <Flipdot bitMatrix={bitMatrix} />
      </div>
      
      {/* Additional info */}
      <div className="mt-4 text-sm text-gray-600">
        <p>Segmentation running at {isConnected ? 'normal' : 'offline'} speed</p>
        <p>Resolution: 36×24 pixels</p>
      </div>
    </div>
  );
}

export default Test;