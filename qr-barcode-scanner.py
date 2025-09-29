import cv2
import numpy as np
from pyzbar.pyzbar import decode
import qrcode
data = 20
qr = qrcode.QRCode(
    version=1,  
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10, 
    border=4, 
)
qr.add_data(data)

img_pil = qr.make_image(fill_color="black", back_color="white").convert("RGB")
img = np.array(img_pil)
cv2.imshow("qr",img)


data=decode(img)

for qr in data:
    content = qr.data.decode()
    polygon = [(point.x, point.y) for point in qr.polygon]
print(content)
top_left = (polygon[0][0] - 25, polygon[0][1] - 25)
bottom_right = (polygon[2][0] + 25, polygon[2][1] + 25)

cv2.rectangle(img,top_left,bottom_right,(255,0,255),2)


cv2.imshow("qr-box",img)

cv2.waitKey(0)
cv2.destroyAllWindows()