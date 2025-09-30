import cv2
import numpy as np

# def Canny_Edge(img, weak_th=None, strong_th=None):
#     height, width = img.shape
    
#     gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
#     gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
#     mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

#     mag_max = np.max(mag)
#     if weak_th is None:
#         weak_th = mag_max * 0.1
#     if strong_th is None: 
#         strong_th = mag_max * 0.5

#     for i_x in range(width):
#         for i_y in range(height):
#             grad_ang = ang[i_y, i_x]
#             grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

#             if grad_ang <= 22.5:
#                 neighb_1_x, neighb_1_y = i_x - 1, i_y
#                 neighb_2_x, neighb_2_y = i_x + 1, i_y
#             elif grad_ang <= 67.5:
#                 neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
#                 neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
#             elif grad_ang <= 112.5:
#                 neighb_1_x, neighb_1_y = i_x, i_y - 1
#                 neighb_2_x, neighb_2_y = i_x, i_y + 1
#             elif grad_ang <= 157.5:
#                 neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
#                 neighb_2_x, neighb_2_y = i_x + 1, i_y - 1
#             else:
#                 neighb_1_x, neighb_1_y = i_x - 1, i_y
#                 neighb_2_x, neighb_2_y = i_x + 1, i_y

#             if 0 <= neighb_1_x < width and 0 <= neighb_1_y < height:
#                 if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
#                     mag[i_y, i_x] = 0
#                     continue
#             if 0 <= neighb_2_x < width and 0 <= neighb_2_y < height:
#                 if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
#                     mag[i_y, i_x] = 0

#     ids = np.zeros_like(img)
#     for i_x in range(width):
#         for i_y in range(height):
#             grad_mag = mag[i_y, i_x]
#             if grad_mag < weak_th:
#                 mag[i_y, i_x] = 0
#             elif strong_th > grad_mag >= weak_th:
#                 ids[i_y, i_x] = 1
#             else:
#                 ids[i_y, i_x] = 2
#     return mag


img = cv2.imread("image.png")  
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.GaussianBlur(gray_image, (5, 5), 0) 
edges = cv2.Canny(blurred_img, 50, 150)
# edges = Canny_Edge(blurred_img)
edges = np.uint8(edges)

_, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if contours:
    doc_box = max(contours, key=cv2.contourArea)
    cv2.drawContours(img, [doc_box], -1, (0,255,0), 3)
    peri = cv2.arcLength(doc_box, True)
    approx = cv2.approxPolyDP(doc_box, 0.02 * peri, True)
    print(approx)
    if len(approx) == 4:
        pts = approx.reshape(4, 2)

        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]   

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  
        rect[3] = pts[np.argmax(diff)]  

        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        cv2.imshow("Scanned Document", warp)


cv2.imshow("Edges", edges)
cv2.imshow("Document Contour", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
