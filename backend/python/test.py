import cv2
import cvzone
import numpy as np
import socket
import urllib.request 
from cvzone.SelfiSegmentationModule import SelfiSegmentation

#for recieve img from esp32cam
url='http://192.168.10.162/cam-hi.jpg'
im = None

#for send to esp32
ESP32_IP = "192.168.1.100" 
ESP32_PORT = 80

# เปิดกล้อง
x = 36
y = 24

segmentor = SelfiSegmentation()

while True:
    # Fetch the image from ESP32-CAM
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    im = cv2.imdecode(imgnp, -1)

    if im is None:
        print("Failed to retrieve image from ESP32-CAM")
        continue

    # Remove background
    imgOut = segmentor.removeBG(im, (0, 0, 0), 0.8)

    # แปลงเป็น Grayscale
    imgGray = cv2.cvtColor(imgOut, cv2.COLOR_BGR2GRAY)

    # ทำ Threshold (0 = ดำ, 255 = ขาว)
    _, imgThresh = cv2.threshold(imgGray, 25, 255, cv2.THRESH_BINARY)

    # ✅ ปรับขนาดข้อมูลเป็น 12x8 (ลดความละเอียด)
    imgSmall = cv2.resize(imgThresh, (x, y), interpolation=cv2.INTER_NEAREST)

    # ✅ แปลงค่า 255 → 1
    imgBinary = imgSmall // 255  # เปลี่ยนค่าจาก 0-255 เป็น 0 หรือ 1

    # ✅ ปริ้นค่าพิกเซล (12×8)
    print("Binary Image (0 = black, 1 = white) [12x8]:")
    print(imgBinary)

    # ✅ ขยายภาพกลับไป 480x320 เพื่อแสดงผล
    imgBig = cv2.resize(imgSmall, (480, 320), interpolation=cv2.INTER_NEAREST)

    # แสดงผล
    cv2.imshow("Binary Mask (640x480)", imgBig)

    # กด 'q' เพื่อออกจากลูป
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
