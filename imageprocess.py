# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:08:20 2022

@author: jevans
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import os
from PIL import Image

def processimage(impath,imname,minpoly=75,maxpoly=1000):
    img = Image.open(os.path.join(impath,imname))
    open_cv_image = np.array(img) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    gray = cv.cvtColor(open_cv_image,cv.COLOR_BGR2GRAY)
    
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im = cv.filter2D(gray, -1, kernel)
    ret, thresh = cv.threshold(im,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    
    
    kernel = np.ones((3,3),np.uint8)
    sharpen2 = cv.dilate(thresh,kernel,iterations=3)
    
    ret, thresh = cv.threshold(sharpen2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    contours,hierachy=cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
    #contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv.contourArea)
    x,y,w,h = cv.boundingRect(big_contour)
    
    # draw filled contour on black background
    mask = np.zeros_like(open_cv_image)
    mask = cv.drawContours(mask, [big_contour], 0, (255,255,255), -1)
    
    mask = cv.cvtColor(mask,cv.COLOR_BGR2GRAY)
    
    newim=sharpen2-mask
    
    contours,hierachy=cv.findContours(newim,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    newcontours=[x for x in contours if (cv.contourArea(x)>minpoly) and (cv.contourArea(x)<maxpoly)]    
    
    mask = np.zeros_like(open_cv_image)
    mask = cv.drawContours(mask, newcontours, -1, (255,255,255), -1)
    
    
    
    mask = cv.cvtColor(mask,cv.COLOR_BGR2GRAY)
    masked = cv.bitwise_and(open_cv_image, open_cv_image, mask=mask)
    return masked

mainpath="C:\\Users\\jevans\\Documents\\GitHub\\mm_images"
impath=os.path.join(mainpath,"training","images")
imnames=os.listdir(impath)
allmasks=[]

for imname in imnames:
    allmasks.append(processimage(impath,imname))
    
    
plt.imshow(allmasks[0])



impath2=os.path.join(mainpath,"training","processedimages2")

fig, axs = plt.subplots(5, 2,  figsize=(20, 50))

axs = axs.flatten()
for img, ax in zip(allmasks, axs):
    ax.imshow(img)
    
fig.savefig(os.path.join(mainpath,"allimages.png"))
    

os.makedirs(impath2,exist_ok=True)    
for i in range(len(allmasks)):
    im = Image.fromarray(allmasks[i])
    im.save(os.path.join(impath2,os.path.splitext(imnames[i])[0]+".tif"))

allmasks=[]    
impath=os.path.join(mainpath,"test","images")
imnames=os.listdir(impath)    

for imname in imnames:
    allmasks.append(processimage(impath,imname))

impath2=os.path.join(mainpath,"test","processedimages")

os.makedirs(impath2,exist_ok=True)    
for i in range(len(allmasks)):
    im = Image.fromarray(allmasks[i])
    im.save(os.path.join(impath2,os.path.splitext(imnames[i])[0]+".tif"))
    


