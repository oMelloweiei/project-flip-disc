import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { io } from "socket.io-client";

// Flipdot component to display the 36x24 bit matrix
const Flipdot = () => {
  const rows = 24; // Number of rows
  const cols = 36; // Number of columns

  // Store previous matrix state for comparison
  const prevMatrixRef = useRef(
    Array.from({ length: rows }, () => Array(cols).fill(false))
  );

  // Initialize bit matrix with all 0s (false)
  const [bitMatrix, setBitMatrix] = useState(
    Array.from({ length: rows }, () => Array(cols).fill(false))
  );

  // Socket.IO connection status
  const [isConnected, setIsConnected] = useState(false);

  // Initialize Socket.IO connection
  useEffect(() => {
    const socket = io("http://localhost:5000");

    socket.on("connect", () => {
      console.log("Connected to Flask-SocketIO server");
      setIsConnected(true);
    });

    socket.on("disconnect", () => {
      console.log("Disconnected from Flask-SocketIO server");
      setIsConnected(false);
    });

    socket.on("connect_error", (error) => {
      console.error("Socket.IO connection error:", error);
      setIsConnected(false);
    });

    socket.on("flipdisc_update", (data) => {
      console.log("Received flipdisc_update data");
      const newMatrix = data.matrix; // 24x36 array of 0s and 1s

      // Convert to boolean values (0 -> false, 1 -> true)
      const newStates = newMatrix.map((row) =>
        row.map((val) => val === 1)
      );

      // Compare with previous state to detect changes
      const prevStates = prevMatrixRef.current;
      let hasChanges = false;

      const updatedStates = newStates.map((row, rowIndex) =>
        row.map((newValue, colIndex) => {
          if (newValue !== prevStates[rowIndex][colIndex]) {
            hasChanges = true;
            return newValue;
          }
          return prevStates[rowIndex][colIndex];
        })
      );

      // Update state only if there are changes
      if (hasChanges) {
        console.log("Updating display - changes detected");
        setBitMatrix(updatedStates);
        prevMatrixRef.current = updatedStates;
      } else {
        console.log("No changes detected - skipping update");
      }
    });

    socket.on("error", (data) => {
      console.error("Server error:", data.message);
    });

    // Cleanup on unmount
    return () => {
      socket.disconnect();
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center h-screen p-4 bg-gray-700">
      {/* Connection status indicator */}
      <div className="mb-4 flex items-center">
        <span
          className={`inline-block w-3 h-3 rounded-full mr-2 ${
            isConnected ? "bg-green-500" : "bg-red-500"
          }`}
        ></span>
        <span>
          {isConnected
            ? "Connected to segmentation server"
            : "Disconnected"}
        </span>
      </div>

      {/* Flipdot grid */}
      {bitMatrix.map((row, rowIndex) => (
        <div key={rowIndex} className="flex">
          {row.map((isFlipped, colIndex) => (
            <motion.div
              key={`${rowIndex}-${colIndex}`}
              className="w-[20px] h-[20px] m-[2px] rounded-full cursor-pointer relative"
              initial={false}
              animate={{ rotateY: isFlipped ? 180 : 0 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              style={{ transformStyle: "preserve-3d" }}
            >
              {/* Front Side (Black) */}
              <div
                className="absolute w-full h-full rounded-full bg-black"
                style={{ backfaceVisibility: "hidden" }}
              />
              {/* Back Side (Lime) */}
              <div
                className="absolute w-full h-full rounded-full bg-lime-500"
                style={{
                  transform: "rotateY(180deg)",
                  backfaceVisibility: "hidden",
                }}
              />
            </motion.div>
          ))}
        </div>
      ))}
    </div>
  );
};

export default Flipdot;