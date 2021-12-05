import cv2 as cv
import numpy as np


def rgb2gray(img):
	src_H, src_W = img.shape[:2]
	empty_img = np.zeros((src_H, src_W), np.float32)
	for i in range(src_H):
		for j in range(src_W):
			empty_img[i, j] = img[i][j][0] * 0.59 + img[i][j][1] * 0.11 + img[i][j][2] * 0.3
	return empty_img.astype(np.uint8)


def function(gray_img):
	print(gray_img.shape)
	empty_img = gray_img / 255
	
	empty_img = np.where(empty_img < 0.5, 0, 255).astype(np.uint8)
	return empty_img


if __name__ == '__main__':
	src_img = cv.imread('lenna.png')
	gray_img = rgb2gray(src_img)
	binary_img = function(gray_img)
	cv.imshow('src_img', src_img)
	cv.imshow('gray_img', gray_img)
	cv.imshow('binary_img', binary_img)
	cv.waitKey()
