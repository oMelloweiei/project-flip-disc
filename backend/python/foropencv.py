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

# Device and model configuration
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Model URLs and paths
U2NET_MODEL_HUMAN_URL = "https://drive.usercontent.google.com/download?id=1m_Kgs91b21gayc2XLW0ou8yugAIadWVP&export=download&authuser=0&confirm=t&uuid=a8419776-e15f-4db8-9264-9b1ee874b6dc&at=APcmpoxHSwr96oGKt_nwwT4Lc9qH%3A1744967213122"
U2NET_MODEL_HUMAN_PATH = "u2net_human_seg.pth"

# Download model if not exists
if not os.path.exists(os.path.join(os.getcwd(), U2NET_MODEL_HUMAN_PATH)):
    print("Downloading U2NET model...")
    _ = gdown.download(U2NET_MODEL_HUMAN_URL, U2NET_MODEL_HUMAN_PATH)

#for recieve img from esp32cam
url='http://192.168.10.162/cam-hi.jpg'
im = None

#for send to esp32
ESP32_IP = "192.168.1.100" 
ESP32_PORT = 81

# Initialize and load model
u2net = U2NET(in_ch=3, out_ch=1)
u2net.load_state_dict(torch.load(U2NET_MODEL_HUMAN_PATH, map_location=DEVICE))
u2net.to(DEVICE)
u2net.eval()

# Constants for image processing
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])
# Processing resolution and flip disc resolution
resize_shape = (320, 240)  # Higher resolution for better detection
flipdisc_resolution = (36, 24)  # Final output for flip disc display

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

def export_flipdisc_data(flipdisc_mask):
    """Convert the binary mask to a format suitable for flip disc display"""
    # Flatten the mask to a 1D array (row by row)
    flat_data = (flipdisc_mask.flatten() > 0).astype(np.uint8)
    
    # Convert to bytes format if needed by your flip disc controller
    # This example packs 8 bits per byte
    bytes_data = np.packbits(flat_data)
    
    return bytes_data

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Initialize processor with U2NET model
    processor = U2NETProcessor(model=u2net, device=DEVICE).start()
    
    prev_frame_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
            
        # Add frame to processing queue
        processor.enqueue(frame)
        
        # Get processed result
        result_available, result = processor.read()
        
        if result_available:
            foreground, full_mask, flipdisc_mask, flipdisc_display = result
            
            # Calculate FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            
            # Display FPS
            cv2.putText(foreground, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display results
            cv2.imshow('Original', frame)
            cv2.imshow('Mask', full_mask)
            cv2.imshow('Foreground', foreground)
            cv2.imshow('Flip Disc (36x24)', flipdisc_display)
            
            # Optional: Export data for flip disc display
            flipdisc_data = export_flipdisc_data(flipdisc_mask)
            # Here you would send flipdisc_data to your display hardware
            
            # Uncomment to print binary representation of flip disc display
            # Useful for debugging
            """
            print("\033[H\033[J")  # Clear console (works on UNIX terminals)
            for y in range(24):
                line = ""
                for x in range(36):
                    line += "●" if flipdisc_mask[y, x] > 0 else "○"
                print(line)
            """
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    processor.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()