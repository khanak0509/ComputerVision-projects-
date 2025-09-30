import cv2
import mediapipe as mp
import time 
import numpy as np
ptime =0 
ctime =0 


def Canny_Edge(img, weak_th=None, strong_th=None):
    height, width = img.shape
    
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    mag_max = np.max(mag)
    if weak_th is None:
        weak_th = mag_max * 0.1
    if strong_th is None: 
        strong_th = mag_max * 0.5

    for i_x in range(width):
            for i_y in range(height):
                grad_ang = ang[i_y, i_x]
                grad_ang = abs(
                    grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

                if grad_ang <= 22.5:
                    neighb_1_x, neighb_1_y = i_x - 1, i_y
                    neighb_2_x, neighb_2_y = i_x + 1, i_y
                elif grad_ang > 22.5 and grad_ang <= 67.5:
                    neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
                    neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
                elif grad_ang > 67.5 and grad_ang <= 112.5:
                    neighb_1_x, neighb_1_y = i_x, i_y - 1
                    neighb_2_x, neighb_2_y = i_x, i_y + 1
                elif grad_ang > 112.5 and grad_ang <= 157.5:
                    neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
                    neighb_2_x, neighb_2_y = i_x + 1, i_y - 1
                else:
                    neighb_1_x, neighb_1_y = i_x - 1, i_y
                    neighb_2_x, neighb_2_y = i_x + 1, i_y

                if 0 <= neighb_1_x < width and 0 <= neighb_1_y < height:
                    if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                        mag[i_y, i_x] = 0
                        continue

                if 0 <= neighb_2_x < width and 0 <= neighb_2_y < height:
                    if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                        mag[i_y, i_x] = 0

    ids = np.zeros_like(img)
    for i_x in range(width):
        for i_y in range(height):
            grad_mag = mag[i_y, i_x]
            if grad_mag < weak_th:
                mag[i_y, i_x] = 0
            elif strong_th > grad_mag >= weak_th:
                ids[i_y, i_x] = 1
            else:
                ids[i_y, i_x] = 2
    return mag

capture = cv2.VideoCapture(0)

while True:
    success , img = capture.read()

    if not success:
        break
    img = cv2.flip(img,1)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred_img = cv2.GaussianBlur(img, (5, 5), 0) 
    
    edges = Canny_Edge(gray_image)
    edges = np.uint8(edges)


    contours, hierarchy=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(edges, contours, -1, (0,255,0), 2)
    

    cv2.imshow("edges", edges)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
