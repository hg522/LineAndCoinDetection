# -*- coding: utf-8 -*-
"""
@author: Himanshu Garg
UBID : 50292195
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import math

s = time.time()

UBID = '50292195'; 
np.random.seed(sum([ord(c) for c in UBID]))

def writeImage(name, img):
    path = "output_imgs/" + name
    cv2.imwrite(path,img)
    print("\n****" + name + " saved****\n")

def getProductSum(l1,l2,N):
    return sum([l1[k]*l2[k] for k in range(N)])


def display(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, np.array(img,dtype=np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getLineAccumulator(img,discret,diag):
    lineaccmltr = np.zeros((diag*2,181))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x] == 255:
                getVotes(x,y,discret,lineaccmltr,diag)
    return lineaccmltr

def getCircleAccumulator(img,rmin,rmax):
    mxdiam = min(img.shape)
    circlaccmltr = np.zeros((img.shape[0],img.shape[1],math.floor(mxdiam/2)))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x] == 255:
                getCircleVotes(img,x,y,circlaccmltr,mxdiam,rmin,rmax)
    return circlaccmltr


def getRange(start, stop, step):
    l = []
    st = start
    while st <= stop:
        l.append(st)
        st+=step
    return l

def getVotes(x,y,d,acc,diag):
    thetas = getRange(-90,90,d)
    for ind,theta in enumerate(thetas):
        r = math.floor(x*np.cos(np.radians(theta)) + y*np.sin(np.radians(theta)))
        acc[r+diag][ind]+=1
        

def getCircleVotes(img,x,y,crcacc,mxdiam,rmin,rmax):
    for r in range(rmin,rmax+1):
        if r >= math.floor(mxdiam/2):
            break
        for t in range(0,361):
            x0 = math.floor(x - r*np.cos(np.radians(t)))
            y0 = math.floor(y - r*np.sin(np.radians(t)))
            if (x0 - r < 0) or (y0 - r < 0) or (x0 + r > img.shape[1]-1) or (y0 + r > img.shape[0]-1):
                continue
            crcacc[y0][x0][r] += 1
    


def getAllLines(acc,th):
    return np.array(np.where(acc > th)).T

def getAllLinesinRange(acc,th1,th2):
    lines = np.array(np.where((acc >= th1) & (acc <= th2))).T
    lines = lines[:,:10]
    return lines

def getAllCircles(acc,th):
    return np.array(np.where(acc > th)).T

def getVerticalLines(rthetas):
    vert = rthetas[np.where(rthetas[:,1] == 88)]
    return vert

def getDiagnolLines(rthetas):
    diagnol = rthetas[np.where(rthetas[:,1] == 54)]
    return diagnol

def convertToPoints(rthetas,h,w,diag):  
    linepts = []
    for rtheta in rthetas:
        #line equation : ax + by + c = 0
        rtheta[0] = rtheta[0] - diag
        a = np.cos(np.radians(rtheta[1]-90))
        b = np.sin(np.radians(rtheta[1]-90))
        c = -rtheta[0]
        y1 = 0
        x1 = int(-c/a)
        y2 = h
        x2 = int((-b*y2 - c)/a)
        
        linepts.append([[x1,y1],[x2,y2]])
    return np.array(linepts)
        
def drawlines(img,linepts,color):
    for pts in linepts:
        x1 = pts[0][0]
        y1 = pts[0][1]
        x2 = pts[1][0]
        y2 = pts[1][1]
        cv2.line(img, (x1,y1), (x2,y2), color,2)
        
def drawCircles(img,circlects,color):
    for pts in circlects:
        c1 = pts[0]
        c2 = pts[1]
        r = pts[2]
        cv2.circle(img,(c2,c1),r,color,2)


def doPadding(img,pd):
    rimg = np.pad(img,pd,'constant')
    return rimg


def calculateConvolution(kernel, image):
    fimg = []
    N = len(kernel)
    pd = int(np.floor(N/2))
    for indxr,row in enumerate(image):
        if indxr < pd or indxr > len(image) - pd - 1:
            continue
        temp = []
        for indc,col in enumerate(row):
            if indc < pd or indc > len(row) - pd - 1:
                continue
            SUM = 0
            for l in range(N):
                r = image[indxr - pd + l][indc - pd:indc + pd + 1]
                SUM+= getProductSum(r,kernel[l],N)
            temp.append(SUM)
        fimg.append(temp)
    return fimg


def filterLines(rthetas):
    newrtheta = []
    temp = []
    for ind,rtheta in enumerate(rthetas):
        if len(temp) == 0:
            temp.append(rtheta)
            temp = np.array(temp)
            if ind == len(rthetas) - 1:
                rt = temp[math.floor(len(temp)/2)]
                newrtheta.append(rt)
        else:
            if (rtheta[0] - min(temp[:,0]) < 8):
                temp = np.append(temp,[rtheta],axis=0)
                if ind == len(rthetas) - 1:
                    rt = temp[math.floor(len(temp)/2)]
                    newrtheta.append(rt)
            else:
                rt = temp[math.floor(len(temp)/2)]
                newrtheta.append(rt)
                temp=[]
                temp.append(rtheta)
                temp = np.array(temp)
                if ind == len(rthetas) - 1:
                    rt = temp[math.floor(len(temp)/2)]
                    newrtheta.append(rt)
    return np.array(newrtheta)


def filterCircles(allcircles):
    newcircles = []
    temp = []
    for ind,circ in enumerate(allcircles):
        if len(temp) == 0:
            temp.append(circ)
            temp = np.array(temp)
            if ind == len(allcircles) - 1:
                c = temp[math.floor(len(temp)/2)]
                newcircles.append(c)
        else:
            diff = abs(circ - temp[len(temp)-1])
            if diff[0] < 7 and diff[1] < 7:
                temp = np.append(temp,[circ],axis=0)
                if ind == len(allcircles) - 1:
                    c = temp[math.floor(len(temp)/2)]
                    newcircles.append(c)
            else:
                c = temp[math.floor(len(temp)/2)]
                newcircles.append(c)
                temp=[]
                temp.append(circ)
                temp = np.array(temp)
                if ind == len(allcircles) - 1:
                    c = temp[math.floor(len(temp)/2)]
                    newcircles.append(c)
    return np.array(newcircles)
            
   
def dilate(img,el):
    pd = np.int32(np.floor(len(el)/2))
    dImg = np.copy(img)
    for indr,row in enumerate(img):
        if indr < pd or indr > len(img) - pd - 1:
            continue
        for indc,col in enumerate(row):
            if indc < pd or indc > len(row) - pd - 1:
                continue
            if el[pd,pd] == img[indr,indc]:
                dImg[indr-pd:indr+pd+1,indc-pd:indc+pd+1] = 255
    return dImg
        
def erode(img,el):
    pd = np.int32(np.floor(len(el)/2))
    dImg = np.copy(img)
    for indr,row in enumerate(img):
        if indr < pd or indr > len(img) - pd - 1:
            continue
        for indc,col in enumerate(row):
            if indc < pd or indc > len(row) - pd - 1:
                continue
            if not np.array_equal(el,img[indr-pd:indr+pd+1,indc-pd:indc+pd+1]):
                dImg[indr,indc] = 0
    return dImg


def doClosing(img,el):
    cimg = dilate(img,el)
    cimg = erode(cimg,el)
    return cimg


################################### task3.1-3.2 ###############################

houghImg = cv2.imread("original_imgs/hough.jpg",0)
houghImgclr = cv2.imread("original_imgs/hough.jpg",1)
redlines = houghImgclr.copy()
blulines = houghImgclr.copy()
blulines2 = houghImgclr.copy()
circleImg = houghImgclr.copy()

discret = 1
krnl = np.array([[255,255,255],[255,255,255],[255,255,255]])
sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]      #flipFilter:sobel_vertical
sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]      #flipFilter:sobel_horizontal
diag = math.floor(np.sqrt((houghImg.shape[0] ** 2) + (houghImg.shape[1] ** 2)))

#hough_thresh = 130
hough_thresh_small1 = 52
hough_thresh_small2 = 52
hough_thresh = 130
kernel_size = 3
rmin = 21
rmax = 23

himg = cv2.GaussianBlur(houghImg,(3,3),2)
himg = doPadding(himg,1)
himgex = calculateConvolution(sobel_x,himg)
himgey = calculateConvolution(sobel_y,himg)
houghEdgeImg = np.sqrt(np.square(himgex) + np.square(himgey))

houghEdgeImgline = cv2.threshold(houghEdgeImg,100,255,cv2.THRESH_BINARY)[1]


lineaccumulator = getLineAccumulator(houghEdgeImgline,discret,diag)
allrthetas = getAllLines(lineaccumulator,hough_thresh)

rthetas = getVerticalLines(allrthetas)
rthetas = filterLines(rthetas)
print("vertical lines: ",rthetas)
linepts = convertToPoints(rthetas,houghImg.shape[0],houghImg.shape[1],diag)
drawlines(redlines,linepts,(0,0,255))
writeImage("red_line.jpg",redlines)


rthetas = getDiagnolLines(allrthetas)
rthetas = filterLines(rthetas)
print("diagnol lines: ",rthetas)
linepts = convertToPoints(rthetas,houghImg.shape[0],houghImg.shape[1],diag)
drawlines(blulines,linepts,(255,0,0))                                      
writeImage("blue_lines.jpg",blulines)

#################################### task3.3 #################################

houghEdgeImgCirc = dilate(houghEdgeImgline,krnl)
houghEdgeImgCirc = erode(houghEdgeImgCirc,krnl)

circleaccumulator = getCircleAccumulator(houghEdgeImgCirc,rmin,rmax)
allcircles = getAllCircles(circleaccumulator,250)
allcircles = filterCircles(allcircles)
sortedCirc = allcircles[allcircles[:,1].argsort()]
allcircles = filterCircles(sortedCirc)
print("\nCircles: \n",allcircles)
drawCircles(circleImg,allcircles,(0,255,0))
writeImage("coin.jpg",circleImg)
print("Total time: ",time.time()-s)




















