import cv2
import numpy as np
from PIL import Image
import LeNet

imgpath=r'./fbk.jpg'
net=LeNet.load_build_net('./weights/LeNet_epoch49.pth')

def get_cloest_char(img):
    dtext=r'/|\=-'
    pixs=np.sum(img>0)
    return dtext[LeNet.demo(net,[Image.fromarray(img)])] if pixs>5 else ' '

def close_demo(image,size=(5,5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    binary = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,anchor=(-1, -1), iterations=3)
    return binary

image = cv2.imread(imgpath)
#image = cv2.resize(image, (600,400), interpolation = cv2.INTER_AREA)
canny = cv2.Canny(image,100,230)
#canny = close_demo(canny,(3,3))
#canny = cv2.resize(canny, (400,600), interpolation = cv2.INTER_AREA)
cv2.imshow('canny',canny)

contours, hierarchy = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
boximg = np.zeros(canny.shape,np.uint8)
cv2.drawContours(boximg,contours,-1,(255,255,255),1)
cv2.imshow("contours", boximg)

cv2.waitKey()

im_shape=image.shape
print(im_shape)
step=10
counter=0

for y in range(0,im_shape[0],step):
    for x in range(0, im_shape[1], step//2):
        im_block=canny[y:y+step,x:x+step//2]
        cch=get_cloest_char(im_block)
        print(cch,end='')
        '''cv2.imshow('cut',im_block)
        cv2.waitKey()'''
    print()
