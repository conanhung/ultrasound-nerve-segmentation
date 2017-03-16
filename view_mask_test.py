import numpy as np
import cv2

test_array = np.load('imgs_test.npy')
mask_array = np.load('imgs_mask_test.npy')

img_idx = 654

test = test_array[img_idx][0].astype('float32')
mask = mask_array[img_idx][0].astype('float32')
mask = np.array(mask * 255, dtype = np.uint8)
resized_mask = cv2.resize(mask, (test.shape[1],test.shape[0]))

# contour detection in resized_mask
ret,thresh = cv2.threshold(resized_mask,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

# apply the overlay
output = test.copy()
output = cv2.cvtColor(output,cv2.COLOR_GRAY2BGR)

# draw the contour
cv2.drawContours(output, contours, -1, (0,0,255), 3)

cv2.imwrite("binary_image/imgs_test_2_0.png", test)
cv2.imwrite("binary_image/imgs_mask_test_2_0.png", thresh)
cv2.imwrite("binary_image/imgs_mask_and_test_2_0.png", output)
