import cv2
import numpy as np

img = cv2.imread("image copy.png")
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.GaussianBlur(gray_image, (5, 5), 0)
edges = cv2.Canny(blurred_img, 50, 150)

_, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if contours:
    doc_box = max(contours, key=cv2.contourArea)
    cv2.drawContours(img, [doc_box], -1, (0, 255, 0), 3)

    peri = cv2.arcLength(doc_box, True)

    approx = cv2.approxPolyDP(doc_box, 0.02 * peri, True)

    if len(approx) != 4:
        print(f"Got {len(approx)} points, falling back to minAreaRect...")
        rect = cv2.minAreaRect(doc_box)
        box = cv2.boxPoints(rect)
        approx = box.astype(int)  

    pts = approx.reshape(4, 2).astype("float32")

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

    gray_warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(
        gray_warp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    cv2.imshow("Final Scanned", scanned)
    cv2.imshow("Scanned Document", warp)

cv2.imshow("Edges", edges)
cv2.imshow("Document Contour", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
