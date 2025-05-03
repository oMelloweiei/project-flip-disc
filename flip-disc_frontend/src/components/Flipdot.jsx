import React, { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import createREGL from "regl";

const rows = 50;
const cols = 112;

const FlipdotWebGL = () => {
  const canvasRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const matrixRef = useRef(
    Array.from({ length: rows }, () => Array(cols).fill(0))
  );

  const resizeCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
  };

  useEffect(() => {
    const socket = io("http://localhost:5000");

    socket.on("connect", () => setIsConnected(true));
    socket.on("disconnect", () => setIsConnected(false));
    socket.on("flipdisc_update", (data) => {
      matrixRef.current = data.matrix;
    });

    return () => socket.disconnect();
  }, []);

  useEffect(() => {
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    const canvas = canvasRef.current;
    const regl = createREGL({ canvas });

    const drawDots = regl({
      vert: `
        precision mediump float;
        attribute vec2 position;
        attribute float flip;
        varying float vFlip;

        void main() {
          vFlip = flip;
          gl_Position = vec4(position, 0, 1);
          gl_PointSize = 10.0;
        }
      `,
      frag: `
        precision mediump float;
        varying float vFlip;

        void main() {
          vec2 centered = gl_PointCoord - vec2(0.5);
          float dist = length(centered);

          if (dist > 0.5) discard;

          vec3 color = mix(vec3(0.1, 0.1, 0.1), vec3(0.6, 1.0, 0.3), vFlip);
          gl_FragColor = vec4(color, 1.0);
        }
      `,
      attributes: {
        position: () =>
          Array.from({ length: rows * cols }, (_, i) => {
            const row = Math.floor(i / cols);
            const col = i % cols;
            return [
              (col / (cols / 2)) - 1,
              1 - (row / (rows / 2)),
            ];
          }),
        flip: () =>
          Array.from({ length: rows * cols }, (_, i) => {
            const row = Math.floor(i / cols);
            const col = i % cols;
            return matrixRef.current[row][col];
          }),
      },
      count: rows * cols,
      primitive: "points",
    });

    regl.frame(() => {
      regl.clear({ color: [0.1, 0.1, 0.1, 1], depth: 1 });
      drawDots();
    });

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      regl.destroy();
    };
  }, []);

  return (
    <div className="fixed inset-0 z-0">
      <canvas ref={canvasRef} className="block w-full h-full" />
      <div className="absolute top-2 left-2 text-white z-10">
        Status: {isConnected ? "Connected" : "Disconnected"}
      </div>
    </div>
  );
};

export default FlipdotWebGL;
