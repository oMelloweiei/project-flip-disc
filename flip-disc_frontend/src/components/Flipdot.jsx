import React, { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import createREGL from "regl";

const rows = 90;
const cols = 160;
const ANIMATION_DURATION = 300; // Animation duration in ms

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
    Array.from({ length: rows }, () => Array(cols).fill(0))
  );

  // Create a gradient pattern across the dots
  const colorRef = useRef(
    Array.from({ length: rows }, (_, r) =>
      Array.from({ length: cols }, (_, c) => {
        // Use position to create a gradient pattern
        const gradientFactor = (r / rows + c / cols) / 2; // 0 to 1 value based on position

        // Choose base color from leaf palette
        const baseColorIndex = Math.floor(Math.random() * leafColors.length);
        const baseColor = leafColors[baseColorIndex];

        // Add slight variations to create natural looking leaves
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

  const resizeCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
  };

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
      // Clear data on disconnect
      matrixRef.current = Array.from({ length: rows }, () => Array(cols).fill(0));
      lastUpdateTimeRef.current = Array.from({ length: rows }, () => Array(cols).fill(0));
    });

    socketRef.current.on("flipdisc_update", (data) => {
      if (data && data.matrix) {
        const now = performance.now();

        // Update animation start times for dots that changed state
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            if (data.matrix[r][c] !== matrixRef.current[r][c]) {
              lastUpdateTimeRef.current[r][c] = now;

              // Generate new leaf color when dot flips on
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

        matrixRef.current = data.matrix;
      }
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  useEffect(() => {
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    const canvas = canvasRef.current;
    if (!canvas) return;

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
          varying vec2 vPointCoord;
          
          void main() {
            vFlip = flip;
            vAnimationProgress = animationProgress;
            vDotColor = dotColor;
            gl_Position = vec4(position, 0, 1);
            gl_PointSize = 10.0;
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
            
            // Flip animation with bounce effect
            float t = vAnimationProgress;
            float animEffect = sin(t * 3.14);
            
            // Create leaf vein patterns
            float veinPattern = 0.0;
            
            // Main central vein
            float mainVein = smoothstep(0.05, 0.0, abs(gl_PointCoord.x - 0.5));
            
            // Side veins (branching out from center)
            for (int i = 1; i <= 3; i++) {
              float y = float(i) * 0.2;
              float sideVein = smoothstep(0.03, 0.0, abs(gl_PointCoord.y - y)) * 
                               smoothstep(0.0, 0.5, gl_PointCoord.x);
              veinPattern += sideVein * 0.3;
            }
            
            veinPattern += mainVein * 0.5;
            
            // Color based on flip state and custom leaf color
            vec3 offColor = vec3(0.1, 0.1, 0.1);
            vec3 onColor = vDotColor;
            
            // Apply vein darkening to create leaf pattern
            onColor = onColor * (1.0 - veinPattern * 0.3);
            
            vec3 color = mix(offColor, onColor, vFlip);
            
            // Apply animation effect (flip with slight variation in lighting)
            float brightness = 0.7 + 0.3 * animEffect;
            
            gl_FragColor = vec4(color * brightness, 1.0);
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
          animationProgress: () =>
            Array.from({ length: rows * cols }, (_, i) => {
              const row = Math.floor(i / cols);
              const col = i % cols;

              const timeSinceUpdate = currentTimeRef.current - lastUpdateTimeRef.current[row][col];
              if (timeSinceUpdate < ANIMATION_DURATION) {
                return Math.min(timeSinceUpdate / ANIMATION_DURATION, 1.0);
              }
              return 1.0; // Animation complete
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

      // Animation frame for rendering
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
    <div className="fixed inset-0 z-0">
      <canvas ref={canvasRef} className="block w-full h-full" />
      <div className="absolute top-2 left-2 p-2 bg-black bg-opacity-50 rounded text-white z-10">
        Status: {isConnected ? "Connected" : "Disconnected"}
      </div>
    </div>
  );
};

export default FlipdotWebGL;