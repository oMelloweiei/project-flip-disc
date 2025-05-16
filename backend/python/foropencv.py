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
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')  


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_FP16 = torch.cuda.is_available()
print(f"Using device: {DEVICE}, FP16: {USE_FP16}")

# Model paths
U2NET_MODEL_HUMAN_PATH = "u2net_human_seg.pth"
YOLO_MODEL_PATH = "yolo12n.pt" 

# ESP32 camera URL for image capture
ESP32_CAM_URL = 'http://192.168.10.162/cam-hi.jpg'

# ESP32 flipdisc controller address
ESP32_FLIPDISC_IP = "192.168.1.100" 
ESP32_FLIPDISC_PORT = 81

# Processing resolution and flip disc resolution
INPUT_RESOLUTION = (320, 240)  # Reduced from 640x480 for faster processing
PROCESS_RESOLUTION = (160, 120)  # Reduced for faster U2NET processing
FLIPDISC_RESOLUTION = (160, 90)  # Final output resolution

# Constants
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])
DETECT_INTERVAL = 15  # Detect person every N frames (reduced from 40)
PERSON_CONFIDENCE_THRESHOLD = 0.75  # Slightly reduced for better responsiveness
IDLE_TIMEOUT_SECONDS = 15
ACTIVE_COOLDOWN_SECONDS = 5

# Video playback settings
IDLE_VIDEO_PATH = "video.mp4"
# Set to -1 to play the entire video before looping
IDLE_VIDEO_MAX_FRAMES = -1  

# Initialize transforms once
transforms = T.Compose([
    T.Resize(PROCESS_RESOLUTION),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

# Create HTTP session for reuse
http_session = requests.Session()

# Cache for idle video frames to avoid disk reads
idle_frames_cache = []
video_frame_count = 0  # Total frames in the video

#CAMERA SPEC
FOCAL_LENGTH_PIXELS = 1422.22
REAL_PERSON_HEIGHT_M = 1.7  

def load_models():
    """Load models with optimizations"""
    global u2net, yolo_model
    
    # Load U2NET with optimizations
    u2net = U2NET(in_ch=3, out_ch=1)
    u2net.load_state_dict(torch.load(U2NET_MODEL_HUMAN_PATH, map_location=DEVICE, weights_only=False))
    u2net.to(DEVICE)
    u2net.eval()
    
    # Enable CUDA optimizations if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if USE_FP16:
            # Enable mixed precision
            u2net = u2net.half()
    
    # Load optimized YOLO model
    yolo_model = YOLO(YOLO_MODEL_PATH)
    if DEVICE != "cpu":
        yolo_model.to(DEVICE)
    
    # Optimize for inference
    if hasattr(yolo_model, 'model'):
        for param in yolo_model.model.parameters():
            param.requires_grad = False
    
    return u2net, yolo_model

class FrameBuffer:
    """Efficient frame buffer to avoid memory allocations"""
    def __init__(self, max_size=3):
        self.frames = [None] * max_size
        self.index = 0
        self.max_size = max_size
        
    def add_frame(self, frame):
        self.frames[self.index] = frame
        self.index = (self.index + 1) % self.max_size
        
    def get_latest(self):
        return self.frames[self.index - 1] if self.frames[self.index - 1] is not None else None

class U2NETProcessor:
    def __init__(self, model, device, queue_size=2):
        self.model = model
        self.device = device
        self.model.eval()
        
        self.frame_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue(maxsize=queue_size)
        self.stopped = False
        
        # Pre-allocate tensors for performance
        self.resized_tensor = torch.zeros((1, 3, PROCESS_RESOLUTION[1], PROCESS_RESOLUTION[0]), 
                                         device=device)
        if USE_FP16:
            self.resized_tensor = self.resized_tensor.half()
        
    def start(self):
        Thread(target=self.process_frames, args=(), daemon=True).start()
        return self
    
    def process_frames(self):
        with torch.no_grad():  # Important for inference performance
            while not self.stopped:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    try:
                        # Convert directly to tensor (faster than PIL)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
                        
                        # Normalize in-place
                        for t, m, s in zip(frame_tensor, MEAN, STD):
                            t.sub_(m).div_(s)
                        
                        # Resize directly with OpenCV for speed (avoids PIL)
                        resized = cv2.resize(frame_rgb, PROCESS_RESOLUTION)
                        
                        # Convert to tensor and normalize
                        resized_tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
                        for t, m, s in zip(resized_tensor, MEAN, STD):
                            t.sub_(m).div_(s)
                        
                        # Add batch dimension and move to device
                        image_tensor = resized_tensor.unsqueeze(0).to(self.device)
                        if USE_FP16:
                            image_tensor = image_tensor.half()
                        
                        # Run model
                        results = self.model(image_tensor)
                        
                        # Process mask
                        pred = torch.squeeze(results[0].cpu(), 0).squeeze(0).numpy()
                        pred_norm = self.normPRED(pred)
                        mask = (pred_norm * 255).astype(np.uint8)
                        
                        # Create flip disc resolution mask
                        flipdisc_mask = cv2.resize(mask, FLIPDISC_RESOLUTION)
                        
                        # Binarize for flip disc (0 or 255)
                        _, flipdisc_mask_binary = cv2.threshold(flipdisc_mask, 127, 255, cv2.THRESH_BINARY)
                        
                        # Create a scaled-up version for display (more efficient resize)
                        flipdisc_mask_display = cv2.resize(flipdisc_mask_binary, 
                                                        (FLIPDISC_RESOLUTION[0]*5, FLIPDISC_RESOLUTION[1]*5), 
                                                        interpolation=cv2.INTER_NEAREST)
                        
                        # Apply mask to create foreground (only when needed for display)
                        full_res_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        mask_3d = cv2.cvtColor(full_res_mask, cv2.COLOR_GRAY2BGR)
                        foreground = cv2.bitwise_and(frame, mask_3d)
                        
                        if not self.result_queue.full():
                            self.result_queue.put((foreground, full_res_mask, flipdisc_mask_binary, flipdisc_mask_display))
                    except Exception as e:
                        print(f"Error in U2NET processing: {e}")
                else:
                    # Small sleep to prevent CPU hogging when queue is empty
                    time.sleep(0.005)
    
    @staticmethod
    def normPRED(predicted_map):
        ma = np.max(predicted_map)
        mi = np.min(predicted_map)
        return (predicted_map - mi) / (ma - mi) if ma > mi else predicted_map
    
    def enqueue(self, frame):
        # Drop frames if queue is full rather than waiting
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
    
    def read(self):
        return not self.result_queue.empty(), self.result_queue.get() if not self.result_queue.empty() else None
    
    def stop(self):
        self.stopped = True

def detect_person(frame, confidence_threshold=0.75):
    if frame.mean() < 20:
        return False, 0.0, []
    small_frame = cv2.resize(frame, (320, 240))
    results = yolo_model(small_frame, verbose=False)[0]
    highest_confidence = 0.0
    person_boxes = []
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())
            if cls_id == 0 and confidence >= confidence_threshold:
                highest_confidence = max(highest_confidence, confidence)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Scale bounding box back to INPUT_RESOLUTION
                scale_x = INPUT_RESOLUTION[0] / 320
                scale_y = INPUT_RESOLUTION[1] / 240
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                box_height = y2 - y1
                if box_height > 0:
                    distance_m = (REAL_PERSON_HEIGHT_M * FOCAL_LENGTH_PIXELS) / box_height
                else:
                    distance_m = float('inf')
                person_boxes.append((x1, y1, x2, y2, box_height, distance_m))
    return len(person_boxes) > 0, highest_confidence, person_boxes

class IdleVideoPlayer:
    """A class to handle idle video playback that supports both memory caching and direct playback"""
    def __init__(self, video_path, input_resolution=(320, 240), max_frames_in_memory=-1):
        self.video_path = video_path
        self.input_resolution = input_resolution
        self.max_frames_in_memory = max_frames_in_memory
        self.frames_cache = []
        self.frame_index = 0
        self.video_capture = None
        
        # Check if the video exists
        if not os.path.exists(video_path):
            print(f"Warning: Video file '{video_path}' not found")
            # Create a blank frame as fallback
            self.frames_cache = [np.zeros((input_resolution[1], input_resolution[0], 3), dtype=np.uint8)]
            return
            
        # Try to open the video
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            print(f"Error: Could not open video file '{video_path}'")
            self.frames_cache = [np.zeros((input_resolution[1], input_resolution[0], 3), dtype=np.uint8)]
            return
            
        # Get video properties
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        print(f"Idle video contains {self.total_frames} frames at {self.fps} FPS")
        
        # If max_frames_in_memory is negative or exceeds the video length, use all frames
        if self.max_frames_in_memory < 0 or self.max_frames_in_memory > self.total_frames:
            self.max_frames_in_memory = self.total_frames
        
        # Load frames into memory if max_frames_in_memory > 0
        if self.max_frames_in_memory > 0:
            self._load_frames_to_memory()
        else:
            # Just make sure we can read the video
            ret, _ = self.video_capture.read()
            if not ret:
                print("Warning: Cannot read from video file, creating blank frame")
                self.frames_cache = [np.zeros((input_resolution[1], input_resolution[0], 3), dtype=np.uint8)]
                self.total_frames = 1
            else:
                # Reset video position
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def _load_frames_to_memory(self):
        """Load video frames into memory cache"""
        print(f"Loading up to {self.max_frames_in_memory} frames into memory...")
        self.frames_cache = []
        
        # Start from beginning
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Read frames
        frame_count = 0
        while frame_count < self.max_frames_in_memory:
            ret, frame = self.video_capture.read()
            if not ret:
                break
                
            # Resize to save memory and standardize size
            frame = cv2.resize(frame, self.input_resolution)
            self.frames_cache.append(frame)
            frame_count += 1
            
        print(f"Loaded {len(self.frames_cache)} frames into memory")
        
        # If we couldn't read any frames, create a blank one
        if not self.frames_cache:
            print("Warning: No frames could be read, creating blank frame")
            self.frames_cache = [np.zeros((self.input_resolution[1], self.input_resolution[0], 3), dtype=np.uint8)]
            
    def get_next_frame(self):
        """Get the next frame from either memory cache or video file"""
        if self.frames_cache:
            # Use memory cache
            frame = self.frames_cache[self.frame_index]
            self.frame_index = (self.frame_index + 1) % len(self.frames_cache)
            return frame
        elif self.video_capture and self.video_capture.isOpened():
            # Read directly from video file
            ret, frame = self.video_capture.read()
            if not ret:
                # We've reached the end, reset to beginning
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_capture.read()
                
            if ret:
                frame = cv2.resize(frame, self.input_resolution)
                return frame
                
        # Fallback to blank frame if everything else fails
        return np.zeros((self.input_resolution[1], self.input_resolution[0], 3), dtype=np.uint8)
        
    def get_frame_count(self):
        """Get the total number of frames"""
        if self.frames_cache:
            return len(self.frames_cache)
        return self.total_frames
        
    def get_current_frame_index(self):
        """Get the current frame index"""
        return self.frame_index
        
    def release(self):
        """Release video resources"""
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()

def export_flipdisc_data(flipdisc_mask):
    """Optimized: Convert binary mask to flip disc format"""
    flat_data = (flipdisc_mask.flatten() > 0).astype(np.uint8)
    bytes_data = np.packbits(flat_data)
    return bytes_data

def convert_mask_to_matrix(flipdisc_mask_binary):
    """Optimized: Convert mask to matrix format for React"""
    # Vectorized approach is much faster than nested loops
    return (flipdisc_mask_binary > 0).astype(int).tolist()

def send_to_esp32(data):
    """Send data to ESP32 with timeout and error handling"""
    try:
        url = f"http://{ESP32_FLIPDISC_IP}:{ESP32_FLIPDISC_PORT}/update"
        response = http_session.post(url, data=data, timeout=0.5)  # Reduced timeout
        return response.status_code == 200
    except Exception:
        return False  # Silently fail to avoid flooding console

def capture_from_esp32cam():
    """Optimized ESP32-CAM capture with timeout and error handling"""
    try:
        response = http_session.get(ESP32_CAM_URL, timeout=1.0)  # Reduced timeout
        if response.status_code == 200:
            # Faster conversion directly to numpy array
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return True, cv2.resize(img, INPUT_RESOLUTION)  # Resize immediately
        return False, None
    except Exception:
        return False, None

def process_frames():
    global video_frame_count
    
    DEBUG_MODE = True
    
    u2net, yolo_model = load_models()
    
    idle_video = IdleVideoPlayer(
        video_path=IDLE_VIDEO_PATH,
        input_resolution=INPUT_RESOLUTION,
        max_frames_in_memory=IDLE_VIDEO_MAX_FRAMES
    )
    
    video_frame_count = idle_video.get_frame_count()
    processor = U2NETProcessor(model=u2net, device=DEVICE).start()
    
    frame_buffer = FrameBuffer(max_size=3)
    prev_frame_time = 0
    frame_counter = 0
    
    last_person_seen_time = time.time() - ACTIVE_COOLDOWN_SECONDS
    highest_recent_confidence = 0.0
    
    mode = "idle_video"
    blackout_mask = np.ones(INPUT_RESOLUTION[::-1], dtype=np.uint8) * 255
    latest_person_boxes = []
    use_webcam = False
    cap = None
    try:
        ret, frame = capture_from_esp32cam()
        if not ret or frame is None:
            print("Falling back to webcam")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_RESOLUTION[1])
            use_webcam = True
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                return
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.resize(frame, INPUT_RESOLUTION)
            else:
                print("Error: Could not read from webcam.")
                return
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return
    last_socket_emit_time = 0
    SOCKET_UPDATE_INTERVAL = 0.05
    while True:
        frame_counter += 1
        frame = None
        if use_webcam:
            if cap is None or not cap.isOpened():
                print("Webcam not available. Trying to reinitialize...")
                try:
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_RESOLUTION[0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_RESOLUTION[1])
                    else:
                        print("Failed to reinitialize webcam")
                except Exception:
                    pass
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame = cv2.resize(frame, INPUT_RESOLUTION)
        else:
            ret, frame = capture_from_esp32cam()
        if not ret or frame is None:
            frame = idle_video.get_next_frame()
            if not use_webcam and frame_counter % 100 == 0:
                try:
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        use_webcam = True
                        print("Switched to webcam after ESP32-CAM failures")
                except Exception:
                    pass
        if frame is not None:
            frame = cv2.flip(frame, 1)
            frame_buffer.add_frame(frame.copy())
            masked_frame = frame.copy()
            masked_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=blackout_mask)
            if frame_counter % DETECT_INTERVAL == 0:
                try:         
                    person_present, highest_recent_confidence, person_boxes = detect_person(frame, PERSON_CONFIDENCE_THRESHOLD)
                    latest_person_boxes = person_boxes

                    person_in_range = False
                    min_distance = float('inf')
                    for (x1, y1, x2, y2, h, distance) in person_boxes:
                        min_distance = min(min_distance, distance)
                        if distance <= 15.0:
                            person_in_range = True

                    blackout_mask.fill(255)
                    
                    if person_present:
                        if not person_in_range:  # No one is in range
                            blackout_mask.fill(0)
                            person_in_range = False
                        else:  # At least one person is in range
                            for (x1, y1, x2, y2, h, distance) in person_boxes:
                                if distance > 15.0:
                                    cv2.rectangle(blackout_mask, (x1, y1), (x2, y2), 0, -1)

                    if person_present and person_in_range:
                        last_person_seen_time = time.time()
                        if mode != "active":
                            print("Person detected - switching to active mode")
                            mode = "active"
                except Exception as e:
                    print(f"Error in person detection: {e}")
            current_time = time.time()
            time_since_person = current_time - last_person_seen_time
            if time_since_person >= IDLE_TIMEOUT_SECONDS and mode != "idle_video":
                print("Timeout reached - switching to idle mode")
                mode = "idle_video"
            foreground = frame
            full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            flipdisc_mask_binary = np.zeros(FLIPDISC_RESOLUTION[::-1], dtype=np.uint8)
            flipdisc_mask_display = np.zeros((FLIPDISC_RESOLUTION[1]*5, FLIPDISC_RESOLUTION[0]*5), dtype=np.uint8)
            result_available = False
            if mode == "idle_video":
                idle_frame = idle_video.get_next_frame()
                gray_idle = cv2.cvtColor(idle_frame, cv2.COLOR_BGR2GRAY)
                _, binary_mask = cv2.threshold(gray_idle, 75, 255, cv2.THRESH_BINARY)
                flipdisc_mask_binary = cv2.resize(binary_mask, FLIPDISC_RESOLUTION, 
                                               interpolation=cv2.INTER_NEAREST)
                _, flipdisc_mask_binary = cv2.threshold(flipdisc_mask_binary, 75, 255, cv2.THRESH_BINARY)
                flipdisc_mask_display = cv2.resize(flipdisc_mask_binary, 
                                               (FLIPDISC_RESOLUTION[0]*5, FLIPDISC_RESOLUTION[1]*5),
                                               interpolation=cv2.INTER_NEAREST)
                foreground = idle_frame.copy()
                if DEBUG_MODE:
                    cv2.rectangle(foreground, (0, 0), (20, 20), (0, 0, 255), -1)
                result_available = True
            else:
                processor.enqueue(masked_frame)
                result_available, result = processor.read()
                if result_available:
                    foreground, full_mask, flipdisc_mask_binary, flipdisc_mask_display = result
            if result_available and (current_time - last_socket_emit_time) >= SOCKET_UPDATE_INTERVAL:
                last_socket_emit_time = current_time
                new_frame_time = time.time()
                fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 30
                prev_frame_time = new_frame_time
                if DEBUG_MODE or cv2.getWindowProperty('Debug View', cv2.WND_PROP_VISIBLE) > 0:
                    if not cv2.getWindowProperty('Debug View', cv2.WND_PROP_VISIBLE) > 0:
                        cv2.namedWindow('Debug View', cv2.WINDOW_NORMAL)
                        cv2.namedWindow('Flip Disc Preview', cv2.WND_PROP_VISIBLE)
                    mode_color = (0, 255, 0) if mode == "active" else (0, 0, 255)
                    cv2.putText(foreground, f"Mode: {mode}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
                    confidence_color = (0, 255, 0) if highest_recent_confidence >= PERSON_CONFIDENCE_THRESHOLD else (0, 0, 255)
                    cv2.putText(foreground, f"Conf: {highest_recent_confidence:.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, confidence_color, 2)
                    
                    minutes = int(time_since_person // 60)
                    seconds = int(time_since_person % 60)
                    
                    cv2.putText(foreground, f"Time: {minutes}m {seconds}s", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(foreground, f"FPS: {int(fps)}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    for (x1, y1, x2, y2, h, distance) in latest_person_boxes:
                        if distance != float('inf'):
                            # Draw bounding box with color based on distance
                            color = (0, 0, 255) if distance > 2.0 else (0, 255, 0)
                            cv2.rectangle(foreground, (x1, y1), (x2, y2), color, 1)
                            
                            # Display distance above bounding box (in meters)
                            distance_text = f"{distance:.1f}m"
                            cv2.putText(foreground, distance_text, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    if mode == "idle_video":
                        current_frame = idle_video.get_current_frame_index()
                        total_frames = idle_video.get_frame_count()
                        cv2.putText(foreground, f"Idle frame: {current_frame}/{total_frames}", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                    cv2.imshow('Flip Disc Preview', flipdisc_mask_display)
                    cv2.imshow('Debug View', foreground)
                try:
                    flipdisc_data = export_flipdisc_data(flipdisc_mask_binary)
                    # send_to_esp32(flipdisc_data)
                except Exception as e:
                    print(f"Error exporting flip disc data: {e}")
                try:
                    matrix = convert_mask_to_matrix(flipdisc_mask_binary)
                    data = {
                        "matrix": matrix,
                        "mode": mode,
                        "fps": int(fps)
                    }
                    if len(socketio.server.environ) > 0:
                        data.update({
                            "timestamp": current_time,
                            "time_since_person": time_since_person,
                            "confidence": highest_recent_confidence,
                            "video_frame": idle_video.get_current_frame_index() if mode == "idle_video" else 0,
                            "video_total_frames": video_frame_count
                        })
                    socketio.emit('flipdisc_update', data)
                except Exception as e:
                    print(f"Error sending data to clients: {e}")
        else:
            time.sleep(0.01)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    processor.stop()
    idle_video.release()
    if use_webcam and cap is not None:
        cap.release()
    cv2.destroyAllWindows()

@app.route('/status')
def status():
    """API endpoint to check system status"""
    return jsonify({
        "status": "running",
        "device": DEVICE,
        "fp16": USE_FP16,
        "video_frames": video_frame_count,
        "version": "1.2.0"
    })

@app.route('/debug/<int:state>')
def set_debug(state):
    """API endpoint to toggle debug mode"""
    global DEBUG_MODE
    DEBUG_MODE = bool(state)
    return jsonify({"debug_mode": DEBUG_MODE})

if __name__ == "__main__":
    # Start the processing thread
    processing_thread = Thread(target=process_frames, daemon=True)
    processing_thread.start()
    
    # Start the SocketIO server (non-blocking in production)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)