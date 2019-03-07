"""
Filename: manual_selection.py
Usage: Select document boundaries manually 
Author: Shashank Sharma
Reference: https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
"""
import imutils
import cv2
import numpy as np 
from skimage.filters import threshold_local


"""
mouse callback function.
param[0]: image
param[1]: color
param[2]: window_name
param[3]: point 1
param[4]: point 2
param[5]: point 3
param[6]: point 4
param[7]: points count
"""
def mouse_callback(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONUP and param[7] < 4:
		param[param[7] + 3] = (x, y)
		cv2.circle(param[0], (x, y), 5, param[1], -1)
		param[7] += 1
		print((x,y))
	if param[7] >= 4:
		pts = np.array(order_points(param[3:-1]), np.int32)
		pts = pts.reshape((-1,1,2))
		cv2.polylines(param[0],[pts],True,(0,255,255), 3)
	cv2.imshow(param[2],param[0])
"""
Function to order points in order below:
top-left, top-right, bottom-right, bottom-left
"""
def order_points(pts):
	pts = np.array(pts)
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped



image = cv2.imread('images/example_03.jpg')
img_copy = image.copy()

window_name = "left click to draw point, press 'q' after done"
cv2.namedWindow(window_name)
color = (0, 0, 255) #blue
param = [img_copy, color, window_name, (0, 0), (0, 0), (0, 0), (0, 0), 0]
cv2.setMouseCallback(window_name, mouse_callback, param)
cv2.imshow(window_name, img_copy)

cv2.waitKey(0)

cv2.destroyAllWindows()

print(param[3:-1])	

ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
	 
# convert the image to grayscale, blur it
warped = four_point_transform(orig, param[3:-1])
 
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 35, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
 
# show the original and scanned images
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
