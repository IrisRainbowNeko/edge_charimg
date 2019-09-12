import cv2
import numpy as np

def aHash(img):
    # 均值哈希算法
    # 缩放为8*8
    gray = cv2.resize(img, (8, 8))
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s+gray[i, j]
    # 求平均灰度
    avg = s/64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str


def cmpHash(hash1, hash2):
    # Hash值对比
    # 算法中1和0顺序组合起来的即是图片的指纹hash。顺序不固定，但是比较的时候必须是相同的顺序。
    # 对比两幅图的指纹，计算汉明距离，即两个64位的hash值有多少是不一样的，不同的位数越小，图片越相似
    # 汉明距离：一组二进制数据变成另一组数据所需要的步骤，可以衡量两图的差异，汉明距离越小，则相似度越高。汉明距离为0，即两张图片完全一样
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def get_cloest_char(img):
    dtext=r'/|\-()'
    match_list=[]
    img=cv2.resize(img, (30, 60))

    cnts2, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts2==[]:
        return ' '

    for item in dtext:
        dsize=cv2.getTextSize(item,cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        im_text=np.zeros((dsize[0][0]*2,dsize[0][1]),dtype=np.uint8)
        im_text = cv2.resize(im_text, (30, 60))
        cv2.putText(im_text, item, (0,dsize[1]+dsize[0][0]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cnts1, hierarchy = cv2.findContours(im_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        '''cv2.imshow('text',im_text)
        cv2.imshow('cut', img)
        cv2.waitKey()'''
        match_list.append(cmpHash(aHash(img),aHash(im_text)))
        #match_list.append(cv2.matchShapes(cnts1[0],cnts2[0],1,0))
    #print(match_list)
    if min(match_list)>18:
        return ' '
    return dtext[np.argmin(match_list)]


image = cv2.imread(r'./hinata.png')
canny = cv2.Canny(image,30,100)

cv2.imshow('canny',canny)
cv2.waitKey()

im_shape=image.shape
print(im_shape)
step=10
counter=0

for y in range(0,im_shape[0],step):
    for x in range(0, im_shape[1], step//2):
        im_block=canny[y:y+step,x:x+step//2]
        cv2.imwrite(f'./imgs/img1_{counter}.png',im_block)
        print(counter)
        counter+=1
        #cch=get_cloest_char(im_block)
        #print(cch,end='')
        '''cv2.imshow('cut',im_block)
        cv2.waitKey()'''
    print('ok')
