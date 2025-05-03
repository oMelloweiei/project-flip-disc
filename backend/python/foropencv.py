import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from threading import Thread
from queue import Queue
import time
import os
import gdown
from u2net import U2NET
from ultralytics import YOLO
import requests
from flask import Flask, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

# Create Flask app with SocketIO
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Device and model configuration
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Model URLs and paths
U2NET_MODEL_HUMAN_PATH = "u2net_human_seg.pth"
yolo_model = YOLO("yolo11n.pt") 

# ESP32 camera URL for image capture
ESP32_CAM_URL = 'http://192.168.10.162/cam-hi.jpg'

# ESP32 flipdisc controller address
ESP32_FLIPDISC_IP = "192.168.1.100" 
ESP32_FLIPDISC_PORT = 81

# Initialize and load model
u2net = U2NET(in_ch=3, out_ch=1)
#use weight only = false for make it work
u2net.load_state_dict(torch.load(U2NET_MODEL_HUMAN_PATH, map_location=DEVICE, weights_only=False))
u2net.to(DEVICE)
u2net.eval()

# Constants for image processing
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])
# Processing resolution and flip disc resolution
resize_shape = (180, 120)  # Higher resolution for better detection
flipdisc_resolution = (112, 50)  # Final output for flip disc display

transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])



class U2NETProcessor:
    def __init__(self, model, device, queue_size=2):
        self.model = model
        self.device = device
        self.model.eval()
        
        self.frame_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue(maxsize=queue_size)
        self.stopped = False
        
    def start(self):
        Thread(target=self.process_frames, args=()).start()
        return self
    
    def process_frames(self):
        while not self.stopped:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                # Convert to PIL Image
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Resize for processing (use a decent size for better detection)
                image_resize = pil_image.resize(resize_shape, resample=Image.BILINEAR)
                
                # Transform
                image_tensor = transforms(image_resize).unsqueeze(0).to(self.device)
                
                # Get prediction
                with torch.no_grad():
                    results = self.model(image_tensor)
                
                # Process mask
                pred = torch.squeeze(results[0].cpu(), dim=(0,1)).numpy()
                pred_norm = self.normPRED(pred)
                mask = (pred_norm * 255).astype(np.uint8)
                
                # Create full resolution mask
                full_res_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                
                # Create flip disc resolution mask (36x24)
                # First resize to target resolution
                flipdisc_mask = cv2.resize(mask, flipdisc_resolution)
                
                # Binarize for flip disc (0 or 255)
                _, flipdisc_mask_binary = cv2.threshold(flipdisc_mask, 127, 255, cv2.THRESH_BINARY)
                
                # Create a scaled-up version of the flip disc mask for display
                flipdisc_mask_display = cv2.resize(flipdisc_mask_binary, (36*10, 24*10), 
                                                  interpolation=cv2.INTER_NEAREST)
                
                # Apply mask to create foreground
                mask_3d = cv2.cvtColor(full_res_mask, cv2.COLOR_GRAY2BGR)
                foreground = cv2.bitwise_and(frame, mask_3d)
                
                if not self.result_queue.full():
                    self.result_queue.put((foreground, full_res_mask, flipdisc_mask_binary, flipdisc_mask_display))
    
    @staticmethod
    def normPRED(predicted_map):
        ma = np.max(predicted_map)
        mi = np.min(predicted_map)
        return (predicted_map - mi) / (ma - mi)
    
    def enqueue(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
    
    def read(self):
        return not self.result_queue.empty(), self.result_queue.get() if not self.result_queue.empty() else None
    
    def stop(self):
        self.stopped = True

def detect_person(frame, confidence_threshold=0.80):
    """
    Detect if a person is present in the frame with confidence above threshold
    
    Args:
        frame: Input image frame
        confidence_threshold: Minimum confidence score to consider a valid detection (0.0-1.0)
        
    Returns:
        Boolean: True if person detected with confidence > threshold, False otherwise
    """
    # Run YOLO detection
    results = yolo_model(frame)[0]
    
    # Check for person class (class 0 in COCO dataset) with confidence > threshold
    person_detected = False
    highest_confidence = 0.0
    
    for r in results.boxes:
        cls_id = int(r.cls.item())
        confidence = float(r.conf.item())  # Get the confidence score
        
        if cls_id == 0:  # class 0 = 'person' in COCO
            highest_confidence = max(highest_confidence, confidence)
            if confidence >= confidence_threshold:
                person_detected = True
                break
    
    # Print detection info
    if highest_confidence > 0:
        print(f"Person detection confidence: {highest_confidence:.2f} "
              f"({'DETECTED' if person_detected else 'BELOW THRESHOLD'})")
    
    return person_detected


def export_flipdisc_data(flipdisc_mask):
    """Convert the binary mask to a format suitable for flip disc display"""
    # Flatten the mask to a 1D array (row by row)
    flat_data = (flipdisc_mask.flatten() > 0).astype(np.uint8)
    
    # Convert to bytes format if needed by your flip disc controller
    # This example packs 8 bits per byte
    bytes_data = np.packbits(flat_data)
    
    return bytes_data

def convert_mask_to_matrix(flipdisc_mask_binary):
    """Convert binary mask to 2D array of 0s and 1s for React"""
    matrix = []
    for y in range(flipdisc_mask_binary.shape[0]):
        row = []
        for x in range(flipdisc_mask_binary.shape[1]):
            # Convert 255 to 1, keep 0 as 0
            row.append(1 if flipdisc_mask_binary[y, x] > 0 else 0)
        matrix.append(row)
    return matrix

def send_to_esp32(data):
    """Send data to ESP32 flipdisc controller"""
    try:
        url = f"http://{ESP32_FLIPDISC_IP}:{ESP32_FLIPDISC_PORT}/update"
        response = requests.post(url, data=data, timeout=1)
        if response.status_code == 200:
            print("Data sent to ESP32 successfully")
        else:
            print(f"Failed to send data: {response.status_code}")
    except Exception as e:
        print(f"Error sending data to ESP32: {e}")

def capture_from_esp32cam():
    """Capture image from ESP32-CAM"""
    try:
        response = requests.get(ESP32_CAM_URL, timeout=5)
        if response.status_code == 200:
            # Convert to numpy array and decode image
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return True, img
        else:
            print(f"Failed to get image from ESP32-CAM: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"Error capturing from ESP32-CAM: {e}")
        return False, None

# SocketIO connection events
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def process_frames():
    """Main processing loop that will emit data to connected clients"""
    # Initialize processor with U2NET model (only used in active mode)
    processor = U2NETProcessor(model=u2net, device=DEVICE).start()

    # Load idle video
    idle_video = cv2.VideoCapture('idle.mp4')
    if not idle_video.isOpened():
        print("Error: Could not open idle video. Make sure the file exists.")
        # Create a black frame as fallback
        idle_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    prev_frame_time = 0

    # Initialize camera settings
    use_webcam = False
    cap = None

    frame_counter = 0
    detect_every_n = 40

    # Idle timeout (30 seconds for testing, was 7 minutes)
    idle_timeout_seconds = 30

    # Cooldown for short absence (5 seconds)
    active_cooldown_seconds = 5

    # Confidence threshold for person detection (80%)
    person_confidence_threshold = 0.80

    last_person_seen_time = time.time() - active_cooldown_seconds  # Start in idle mode
    highest_recent_confidence = 0.0  # Track highest recent confidence for display

    try:
        # First try ESP32-CAM
        ret, frame = capture_from_esp32cam()
        if not ret or frame is None:
            # Fallback to webcam
            print("Falling back to webcam")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            use_webcam = True

            if not cap.isOpened():
                print("Error: Could not open webcam.")
                return

            # Test webcam
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Could not read from webcam.")
                return
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return

    while True:
        frame_counter += 1
        frame = None

        # Get frame from either ESP32-CAM or webcam
        if use_webcam:
            if cap is None or not cap.isOpened():
                print("Webcam not available. Trying to reinitialize...")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Failed to reinitialize webcam. Exiting...")
                    break
            ret, frame = cap.read()
        else:
            ret, frame = capture_from_esp32cam()

        if not ret or frame is None:
            print("Error: Failed to capture frame.")
            # If ESP32-CAM fails, try to switch to webcam
            if not use_webcam:
                print("Switching to webcam")
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    use_webcam = True
                    continue
                else:
                    print("Failed to open webcam as fallback")

            # If all camera options fail, use the idle video frame
            ret_idle, idle_frame = idle_video.read()
            if ret_idle and idle_frame is not None:
                frame = idle_frame
            else:
                # Reset video to beginning if at end
                idle_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_idle, idle_frame = idle_video.read()
                if ret_idle and idle_frame is not None:
                    frame = idle_frame
                else:
                    # Last resort - create a black frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Flip frame horizontally
        if frame is not None:
            frame = cv2.flip(frame, 1)

            # Detect person every N frames
            if frame_counter % detect_every_n == 0:
                try:
                    # Get person detection with confidence check
                    person_present = detect_person(frame, person_confidence_threshold)

                    # Get the raw results to extract confidence for display
                    raw_results = yolo_model(frame)[0]
                    highest_confidence = 0.0

                    for r in raw_results.boxes:
                        cls_id = int(r.cls.item())
                        confidence = float(r.conf.item())
                        if cls_id == 0:  # Person class
                            highest_confidence = max(highest_confidence, confidence)

                    highest_recent_confidence = highest_confidence

                    if person_present:
                        last_person_seen_time = time.time()
                except Exception as e:
                    print(f"Error in person detection: {e}")
                    person_present = False
                    highest_recent_confidence = 0.0

            current_time = time.time()
            time_since_person = current_time - last_person_seen_time

            # Initialize output variables
            foreground = frame  # Default to input frame
            full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)  # Default empty mask
            flipdisc_mask_binary = np.zeros(flipdisc_resolution[::-1], dtype=np.uint8)  # Default empty flip disc mask
            flipdisc_mask_display = cv2.resize(flipdisc_mask_binary, (flipdisc_resolution[0]*10, flipdisc_resolution[1]*10),
                                              interpolation=cv2.INTER_NEAREST)

            if time_since_person >= idle_timeout_seconds:
                # Idle mode: Use idle video and create binary mask without segmentation
                mode = "idle_video"
                ret_idle, idle_frame = idle_video.read()
                if not ret_idle or idle_frame is None:
                    idle_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop idle video
                    ret_idle, idle_frame = idle_video.read()
                    if not ret_idle or idle_frame is None:
                        idle_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Fallback to black frame

                # Create binary mask from idle frame (simple thresholding)
                gray_idle = cv2.cvtColor(idle_frame, cv2.COLOR_BGR2GRAY)
                _, binary_mask = cv2.threshold(gray_idle, 127, 255, cv2.THRESH_BINARY)
                
                # Resize to flip disc resolution
                flipdisc_mask_binary = cv2.resize(binary_mask, flipdisc_resolution, interpolation=cv2.INTER_NEAREST)
                _, flipdisc_mask_binary = cv2.threshold(flipdisc_mask_binary, 127, 255, cv2.THRESH_BINARY)
                
                # Create display version
                flipdisc_mask_display = cv2.resize(flipdisc_mask_binary, (flipdisc_resolution[0]*10, flipdisc_resolution[1]*10),
                                                  interpolation=cv2.INTER_NEAREST)
                
                # Use idle frame as foreground for visualization
                foreground = idle_frame
                full_mask = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                result_available = True
            else:
                # Active mode: Use U2NET segmentation
                mode = "active"
                processor.enqueue(frame)
                
                # Get processed result
                result_available, result = processor.read()
                
                if result_available:
                    foreground, full_mask, flipdisc_mask_binary, flipdisc_mask_display = result
                else:
                    # If no result yet, use defaults and skip processing
                    result_available = False

            if result_available:
                # Calculate FPS
                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 30
                prev_frame_time = new_frame_time

                # Display mode on output
                mode_color = (0, 255, 0) if mode == "active" else (0, 0, 255)
                cv2.putText(foreground, f"Mode: {mode}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)

                # Display confidence level
                confidence_color = (0, 255, 0) if highest_recent_confidence >= person_confidence_threshold else (0, 0, 255)
                cv2.putText(foreground, f"Confidence: {highest_recent_confidence:.2f}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, confidence_color, 2)

                # Display time since last person detected
                minutes = int(time_since_person // 60)
                seconds = int(time_since_person % 60)
                cv2.putText(foreground, f"Time since person: {minutes}m {seconds}s", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display flip disc preview
                try:
                    cv2.imshow('Flip Disc Preview', flipdisc_mask_display)
                    cv2.imshow('Debug View', foreground)
                except Exception as e:
                    print(f"Error displaying preview: {e}")

                # Export data for flip disc display
                try:
                    flipdisc_data = export_flipdisc_data(flipdisc_mask_binary)
                    # Uncomment to send to actual hardware:
                    # send_to_esp32(flipdisc_data)
                except Exception as e:
                    print(f"Error exporting flip disc data: {e}")

                # Convert to matrix format for React
                try:
                    matrix = convert_mask_to_matrix(flipdisc_mask_binary)

                    # Emit data to connected clients via SocketIO
                    data = {
                        "matrix": matrix,
                        "timestamp": current_time,
                        "fps": int(fps),
                        "mode": mode,
                        "time_since_person": time_since_person,
                        "confidence": highest_recent_confidence
                    }
                    socketio.emit('flipdisc_update', data)
                except Exception as e:
                    print(f"Error sending data to clients: {e}")
        else:
            print("Warning: Null frame encountered")
            time.sleep(0.1)  # Small delay to prevent tight loop

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    processor.stop()
    if use_webcam and cap is not None:
        cap.release()
    idle_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start the processing thread
    processing_thread = Thread(target=process_frames)
    processing_thread.daemon = True
    processing_thread.start()
    
    # Start the SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)