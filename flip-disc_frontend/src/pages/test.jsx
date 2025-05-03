// src/pages/FlipdotDisplay.js
import React from "react";
import Flipdot from "../components/Flipdot"; // Adjust the path based on your file structure

const FlipdotDisplay = () => {
  return (
    <div className="min-h-screen text-white ">
      <div className="max-w mx-auto">
        {/* Page Header */}
        <h1 className="text-3xl font-bold mb-4 text-center">
          FlipDisc Simulation
        </h1>
        <p className="text-lg mb-6 text-center text-gray-300">
          {/* This page displays a 36x24 flip-dot simulation driven by real-time human
          segmentation data from a U2NET model running on a Flask server. */}
        </p>

        {/* Flipdot Component */}
        <div className="flex justify-center">
          <Flipdot />
        </div>

        {/* Additional Info */}
        <div className="mt-6 text-sm text-gray-400 text-center">
          <p>Resolution: 36×24 pixels</p>
          <p>
            Powered by Flask-SocketIO and U2NET model processing ESP32-CAM images
          </p>
        </div>
      </div>
    </div>
  );
};

export default FlipdotDisplay;