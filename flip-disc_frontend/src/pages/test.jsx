import React from "react";
import Flipdot from "../components/Flipdot";

const bitMatrix = Array.from({ length: 24 }, () =>
  Array.from({ length: 36 }, () => Math.round(Math.random())) // Random 0 or 1
);

function Test() {
  return (
    <div>
      <Flipdot />
    </div>
  );
}

export default Test;
