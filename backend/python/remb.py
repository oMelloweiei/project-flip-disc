import cv2
import rembg
import numpy as np
import io

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    if not success:
        continue

    # Convert image to byte format for rembg
    _, img_bytes = cv2.imencode('.png', img)
    img_byte_array = img_bytes.tobytes()

    # Remove background using rembg
    output_byte_array = rembg.remove(img_byte_array)

    # Convert the output back to a numpy array (OpenCV format)
    output_array = np.frombuffer(output_byte_array, np.uint8)
    output_img = cv2.imdecode(output_array, cv2.IMREAD_UNCHANGED)

    # Display the result
    cv2.imshow("Background Removed", output_img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
