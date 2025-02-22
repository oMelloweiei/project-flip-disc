import cv2
import cvzone
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation

# เปิดกล้อง
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

segmentor = SelfiSegmentation()

while True:
    success, img = cap.read()
    if not success:
        continue

    # ลบพื้นหลัง
    imgOut = segmentor.removeBG(img, (0, 0, 0), 0.8)

    # แปลงเป็น Grayscale
    imgGray = cv2.cvtColor(imgOut, cv2.COLOR_BGR2GRAY)

    # ทำ Threshold (0 = ดำ, 255 = ขาว)
    _, imgThresh = cv2.threshold(imgGray, 25, 255, cv2.THRESH_BINARY)

    # ✅ ปรับขนาดข้อมูลเป็น 12x8 (ลดความละเอียด)
    imgSmall = cv2.resize(imgThresh, (12, 8), interpolation=cv2.INTER_NEAREST)

    # ✅ แปลงค่า 255 → 1
    imgBinary = imgSmall // 255  # เปลี่ยนค่าจาก 0-255 เป็น 0 หรือ 1

    # ✅ ปริ้นค่าพิกเซล (12×8)
    print("Binary Image (0 = black, 1 = white) [12x8]:")
    print(imgBinary)

    # ✅ ขยายภาพกลับไป 640×480 เพื่อแสดงผล
    imgBig = cv2.resize(imgSmall, (640, 480), interpolation=cv2.INTER_NEAREST)

    # แสดงผล
    cv2.imshow("Binary Mask (640x480)", imgBig)

    # กด 'q' เพื่อออกจากลูป
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
