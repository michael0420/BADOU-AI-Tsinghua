import cv2
from skimage import util
img = cv2.imread('F:/lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = util.random_noise(gray, 'gaussian')
img2 = util.random_noise(gray, 's&p')
cv2.imshow('img', img)
cv2.imshow('gray', gray)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey(0)