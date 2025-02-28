import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";

// Flipdot component to display the bitmatrix
const Flipdot = () => {
    const rows = 24;  // Number of rows
    const cols = 36;  // Number of columns

    // Store previous state for comparison
    const prevStateRef = useRef(
        Array.from({ length: rows }, () => Array(cols).fill(false))
    );

    // Initialize the bitmatrix with all 0s
    const [flippedStates, setFlippedStates] = useState(
        Array.from({ length: rows }, () => Array(cols).fill(false))
    );

    useEffect(() => {
        const socket = new WebSocket("ws://localhost:8765");

        socket.onopen = () => {
            console.log("WebSocket connection established");
        };

        socket.onmessage = (event) => {
            console.log("Data received from server");
            const bitMatrix = JSON.parse(event.data);

            // Convert incoming matrix to boolean values
            const newStates = bitMatrix.map(row => row.map(val => val === 1));

            // Compare with previous state and only update dots that changed
            const prevStates = prevStateRef.current;
            let hasChanges = false;

            // Create a copy of the current state
            const updatedStates = newStates.map((row, rowIndex) =>
                row.map((newValue, colIndex) => {
                    // Check if this specific dot has changed
                    if (newValue !== prevStates[rowIndex][colIndex]) {
                        hasChanges = true;
                        return newValue;
                    }
                    // No change for this dot, keep previous state
                    return prevStates[rowIndex][colIndex];
                })
            );

            // Only update state if there were changes
            if (hasChanges) {
                console.log("Display updated - changes detected");
                setFlippedStates(updatedStates);
                prevStateRef.current = updatedStates;
            } else {
                console.log("No changes detected - skipping update");
            }
        };

        socket.onerror = (error) => {
            console.error("WebSocket error:", error);
        };

        socket.onclose = () => {
            console.log("WebSocket connection closed");
        };

        return () => {
            socket.close();
        };
    }, []);

    return (
        <div className="flex flex-col items-center justify-center h-screen p-4 bg-gray-700">
            {flippedStates.map((row, rowIndex) => (
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
                            {/* Front Side */}
                            <div className="absolute w-full h-full rounded-full bg-black"
                                style={{ backfaceVisibility: "hidden" }}
                            />
                            {/* Back Side */}
                            <div className="absolute w-full h-full rounded-full bg-lime-500"
                                style={{ transform: "rotateY(180deg)", backfaceVisibility: "hidden" }}
                            />
                        </motion.div>
                    ))}
                </div>
            ))}
        </div>
    );
};

export default Flipdot;