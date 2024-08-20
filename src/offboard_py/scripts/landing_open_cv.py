#!/usr/bin/env python3
import cv2
import argparse
import sys
import numpy as np
import math


# if you want to use the video,,, path=~ and attribute path instead of 0
def find_H_shape(img):
    # Gaussian Pre-Processing
    blurred=cv2.GaussianBlur(img, (5,5),0)
    gray=cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray, 50,150, apertureSize=3)
    
    # hough transform
    lines=cv2.HoughLinesP(edges, 1, np,pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2=line[0]
            cv2.line(img, (x1, y1), (x2,y2), (0,255,0),2)
    
    # H sahpe... 
    return img

img=cv2.imread('H.jpg')
result_img=find_H_shape(img)

cv2.imshow('Detected H Shape', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

