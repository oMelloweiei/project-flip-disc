import React from "react";

function Test() {
  return (
    <div>
      <h1 className="text-white">Test Page</h1>
      <div className="video-container">
        <img
          src="https://localhost:8080/webcam"
          alt="Video Stream"
          className="streaming-video"
        />
      </div>
    </div>
  );
}

export default Test;
