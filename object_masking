import cv2
import numpy as np

img = cv2.imread("motherboard_image.JPEG")
original = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
edges = cv2.Canny(gray, 80, 200)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

mask = np.zeros_like(gray)
cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

extracted = cv2.bitwise_and(original, original, mask=mask)

cv2.imwrite("edges_output.png", edges)
cv2.imwrite("mask_output.png", mask)
cv2.imwrite("extracted_output.png", extracted)

print("Masking complete")
