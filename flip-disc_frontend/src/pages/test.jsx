// src/pages/FlipdotDisplay.js
import React from "react";
import FlipdotWebGL from "../components/Flipdot"; // Adjust the path based on your file structure

const FlipdotDisplay = () => {
  return (
          <FlipdotWebGL rows={72} cols={128} />
  );
};

export default FlipdotDisplay;