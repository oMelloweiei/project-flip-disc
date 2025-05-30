import React, { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import createREGL from "regl";

const rows = 45;
const cols = 80;
const ANIMATION_DURATION = 100; // Animation duration in ms

// Leaf-inspired color palette
const leafColors = [
  [0.20, 0.76, 0.25], // Fresh green
  [0.13, 0.55, 0.13], // Medium green
  [0.00, 0.39, 0.00], // Dark green
  [0.56, 0.93, 0.56], // Light green
  [0.48, 0.77, 0.46], // Seafoam green
  [0.67, 0.84, 0.29], // Yellow-green
  [0.33, 0.42, 0.18], // Olive green
  [0.19, 0.50, 0.08], // Forest green
  [0.38, 0.61, 0.25], // Grass green
  [0.29, 0.70, 0.33]  // Spring green
];

const FlipdotWebGL = () => {
  const canvasRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const matrixRef = useRef(
    Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (Math.random() > 0.7 ? 1 : 0))
    )
  );
  const colorRef = useRef(
    Array.from({ length: rows }, (_, r) =>
      Array.from({ length: cols }, (_, c) => {
        const gradientFactor = (r / rows + c / cols) / 2;
        const baseColorIndex = Math.floor(Math.random() * leafColors.length);
        const baseColor = leafColors[baseColorIndex];
        return [
          baseColor[0] * (0.9 + Math.random() * 0.2),
          baseColor[1] * (0.9 + Math.random() * 0.2),
          baseColor[2] * (0.9 + Math.random() * 0.2)
        ];
      })
    )
  );
  const lastUpdateTimeRef = useRef(
    Array.from({ length: rows }, () => Array(cols).fill(0))
  );
  const socketRef = useRef(null);
  const reglRef = useRef(null);
  const currentTimeRef = useRef(0);

  useEffect(() => {
    // Socket.io connection
    socketRef.current = io("http://localhost:5000");

    socketRef.current.on("connect", () => {
      console.log("Connected to server");
      setIsConnected(true);
    });

    socketRef.current.on("disconnect", () => {
      console.log("Disconnected from server");
      setIsConnected(false);
      matrixRef.current = Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => (Math.random() > 0.7 ? 1 : 0))
      );
      lastUpdateTimeRef.current = Array.from({ length: rows }, () => Array(cols).fill(0));
    });

    socketRef.current.on("flipdisc_update", (data) => {
      if (data && data.matrix) {
        const receivedRows = data.matrix.length;
        const receivedCols = receivedRows > 0 ? data.matrix[0].length : 0;
        console.log(`Received matrix: ${receivedRows}x${receivedCols}`);
        const now = performance.now();
        for (let r = 0; r < Math.min(rows, receivedRows); r++) {
          for (let c = 0; c < Math.min(cols, receivedCols); c++) {
            if (data.matrix[r][c] !== matrixRef.current[r][c]) {
              lastUpdateTimeRef.current[r][c] = now;
              if (data.matrix[r][c] === 1) {
                const baseColorIndex = Math.floor(Math.random() * leafColors.length);
                const baseColor = leafColors[baseColorIndex];
                colorRef.current[r][c] = [
                  baseColor[0] * (0.9 + Math.random() * 0.2),
                  baseColor[1] * (0.9 + Math.random() * 0.2),
                  baseColor[2] * (0.9 + Math.random() * 0.2)
                ];
              }
            }
          }
        }
        const newMatrix = Array.from({ length: rows }, (_, r) =>
          Array.from({ length: cols }, (_, c) =>
            r < receivedRows && c < receivedCols
              ? data.matrix[r][c]
              : matrixRef.current[r][c]
          )
        );
        matrixRef.current = newMatrix;
      } else {
        console.warn("Invalid flipdisc_update data:", data);
      }
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  useEffect(() => {
    // Handle canvas resizing
    const resizeCanvas = () => {
      if (canvasRef.current && reglRef.current) {
        const canvas = canvasRef.current;
        const devicePixelRatio = window.devicePixelRatio || 1;
        canvas.width = Math.floor(window.innerWidth * devicePixelRatio);
        canvas.height = Math.floor(window.innerHeight * devicePixelRatio);
        canvas.style.width = window.innerWidth + "px";
        canvas.style.height = window.innerHeight + "px";
        reglRef.current._gl.viewport(0, 0, canvas.width, canvas.height);
        console.log(`Canvas resized: ${canvas.width}x${canvas.height}`);
      }
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    document.body.style.margin = "0";
    document.body.style.padding = "0";
    document.body.style.overflow = "hidden";

    const canvas = canvasRef.current;
    if (!canvas) {
      console.error("Canvas not found");
      return;
    }

    try {
      reglRef.current = createREGL({ canvas });

      const drawDots = reglRef.current({
        vert: `
          precision mediump float;
          attribute vec2 position;
          attribute float flip;
          attribute float animationProgress;
          attribute vec3 dotColor;
          varying float vFlip;
          varying float vAnimationProgress;
          varying vec3 vDotColor;
          
          void main() {
            vFlip = flip;
            vAnimationProgress = animationProgress;
            vDotColor = dotColor;
            gl_Position = vec4(position, 0, 1);
            float pointSize = min(float(${window.innerWidth}) / ${cols}.0, float(${window.innerHeight}) / ${rows}.0) * 1.2;
            gl_PointSize = max(pointSize, 4.0);
          }
        `,
        frag: `
          precision mediump float;
          varying float vFlip;
          varying float vAnimationProgress;
          varying vec3 vDotColor;
          
          void main() {
            vec2 centered = gl_PointCoord - vec2(0.5);
            float dist = length(centered);
            
            if (dist > 0.5) discard;
            
            float t = vAnimationProgress;
            float animEffect = sin(t * 3.14);
            
            float veinPattern = 0.0;
            float mainVein = smoothstep(0.05, 0.0, abs(gl_PointCoord.x - 0.5));
            for (int i = 1; i <= 3; i++) {
              float y = float(i) * 0.2;
              float sideVein = smoothstep(0.03, 0.0, abs(gl_PointCoord.y - y)) * 
                               smoothstep(0.0, 0.5, gl_PointCoord.x);
              veinPattern += sideVein * 0.3;
            }
            veinPattern += mainVein * 0.5;
            
            vec3 offColor = vec3(0.1, 0.1, 0.1);
            vec3 onColor = vDotColor;
            onColor = onColor * (1.0 - veinPattern * 0.3);
            
            vec3 color = mix(offColor, onColor, vFlip);
            float brightness = 0.7 + 0.3 * animEffect;
            
            gl_FragColor = vec4(color * brightness, 1.0);
          }
        `,
        attributes: {
          position: () => {
            const positions = Array.from({ length: rows * cols }, (_, i) => {
              const row = Math.floor(i / cols);
              const col = i % cols;
              return [
                (col / (cols - 1)) * 2 - 1,
                1 - (row / (rows - 1)) * 2,
              ];
            });
            console.log(`Position array length: ${positions.length}`);
            return positions;
          },
          flip: () => {
            const flips = Array.from({ length: rows * cols }, (_, i) => {
              const row = Math.floor(i / cols);
              const col = i % cols;
              return matrixRef.current[row][col];
            });
            console.log(`Flip array non-zero count: ${flips.filter(x => x !== 0).length}`);
            return flips;
          },
          animationProgress: () =>
            Array.from({ length: rows * cols }, (_, i) => {
              const row = Math.floor(i / cols);
              const col = i % cols;
              const timeSinceUpdate = currentTimeRef.current - lastUpdateTimeRef.current[row][col];
              return Math.min(timeSinceUpdate / ANIMATION_DURATION, 1.0);
            }),
          dotColor: () =>
            Array.from({ length: rows * cols }, (_, i) => {
              const row = Math.floor(i / cols);
              const col = i % cols;
              return colorRef.current[row][col];
            }),
        },
        count: rows * cols,
        primitive: "points",
      });

      const animate = () => {
        if (!reglRef.current) return;
        currentTimeRef.current = performance.now();
        reglRef.current.clear({ color: [0.1, 0.1, 0.1, 1], depth: 1 });
        drawDots();
        requestAnimationFrame(animate);
      };

      const frameId = requestAnimationFrame(animate);

      return () => {
        window.removeEventListener("resize", resizeCanvas);
        cancelAnimationFrame(frameId);
        if (reglRef.current) {
          reglRef.current.destroy();
          reglRef.current = null;
        }
      };
    } catch (error) {
      console.error("Error initializing WebGL:", error);
      return () => {
        window.removeEventListener("resize", resizeCanvas);
      };
    }
  }, []);

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        margin: 0,
        padding: 0,
        overflow: "hidden",
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          display: "block",
          width: "100vw",
          height: "100vh",
        }}
      />
    </div>
  );
};

export default FlipdotWebGL;