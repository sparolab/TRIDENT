"""
# > Modules for computing the Underwater Image Quality Measure (UIQM)
# Maintainer: Jahid (email: islam034@umn.edu)
"""
from scipy import ndimage
from PIL import Image
import numpy as np
import math
from skimage import color

import numpy as np
import cv2

# def uciqe_compute(img):
    
#     img = img.astype(np.float32)
#     hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

#     H,S,V = cv2.split(hsv)
#     delta = np.std(H) #色度的标准方差
#     mu = np.mean(S) #饱和度平均值
#     n,m = img.shape[0:2]
#     number = int(n*m/100)
#     Maxsum = 0
#     Minsum = 0
#     V1 = V
#     V2 = V
#     for i in range(number): #最大像素值，前1/100
#         Maxvaule = np.max(np.max(V1))
#         [x,y] = np.where(V1 == Maxvaule)
#         Maxsum = Maxsum + V1[x[0],y[0]]
#         V1[x[0],y[0]] = 0 #最大值赋0
    
#     top = Maxsum / number

#     for i in range(number): #最小像素值，前1/100
#         Minvaule = np.min(np.min(V2))
#         [x,y] = np.where(V2 == Minvaule)
#         Minsum = Minsum + V2[x[0],y[0]]
#         V2[x[0],y[0]] = 1 #最小值赋1
    
#     buttom = Minsum / number #平均
#     conl = top - buttom 

#     uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu

#     return uciqe


# def uciqe_compute(img):
    
#     a = img.astype(np.uint8)
#     lab = color.rgb2lab(a)
    
#     # UCIQE
#     c1 = 0.4680
#     c2 = 0.2745
#     c3 = 0.2576
#     l = lab[:,:,0]
    
#     l = l / 255.0 + 0.00001
#     a = lab[:,:,1] / 255.0 
#     b = lab[:,:,2] / 255.0 
    
#     #1st term
#     chroma = (a**2 + b**2)**0.5

#     # saturation = chroma / l
#     saturation = chroma / ((chroma**2+l**2)**0.5)
#     aver_sat = np.mean(saturation)
#     aver_chr = np.mean(chroma)
#     var_chr = np.mean(abs(1-((aver_chr / chroma)**2)))**0.5
    
#     contrast_l = l.max() - l.min()
    
#     uciqe = c1 * var_chr + c2 * contrast_l + c3 * aver_sat

#     return uciqe




# def uciqe_compute(img):
    
#     a = img.astype(np.uint8)
#     lab = color.rgb2lab(a)
    
#     # UCIQE
#     c1 = 0.4680
#     c2 = 0.2745
#     c3 = 0.2576
#     l = lab[:,:,0]
    
#     # l = l / 255.0 + 0.00001
#     # a = lab[:,:,1] / 255.0
#     # b = lab[:,:,2] / 255.0

#     l = l + 0.00001
#     a = lab[:,:,1]
#     b = lab[:,:,2]
    
#     #1st term
#     chroma = (a**2 + b**2)**0.5

#     # saturation = chroma / l
#     saturation = chroma / ((chroma**2+l**2)**0.5)
#     aver_sat = np.mean(saturation)
#     aver_chr = np.mean(chroma)
#     var_chr = np.mean((abs(chroma**2 - aver_chr**2)))**0.5
    
#     contrast_l = l.max() - l.min()
    
#     uciqe = c1 * var_chr + c2 * contrast_l + c3 * aver_sat

#     return uciqe



def uciqe_compute(img):

    a = img.astype(np.uint8)
    lab = color.rgb2lab(a) / 255
    # gray = color.rgb2gray(a)
    
    # UCIQE
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    l = lab[:,:,0]

    #1st term
    chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
    uc = np.mean(chroma)
    # sc = (np.mean((chroma - uc)**2))**0.5
    sc = (np.mean((chroma**2 - uc**2)))**0.5

    #2nd term
    top = np.int(np.round(0.01*l.shape[0]*l.shape[1]))
    sl = np.sort(l,axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[:top])-np.mean(sl[:top])

    #3rd term
    satur = []
    chroma1 = chroma.flatten()
    l1 = l.flatten()
    for i in range(len(l1)):
        if chroma1[i] == 0: satur.append(0)
        elif l1[i] == 0: satur.append(0)
        else: satur.append(chroma1[i] / l1[i])

    us = np.mean(satur)

    uciqe = c1 * sc + c2 * conl + c3 * us

    return uciqe




# def uciqe_compute(img):

#     a = img.astype(np.uint8)
#     lab = color.rgb2lab(a) / 255.0
#     # gray = color.rgb2gray(a)
    
#     # UCIQE
#     c1 = 0.4680
#     c2 = 0.2745
#     c3 = 0.2576
#     l = lab[:,:,0]

#     #1st term
#     chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
#     uc = np.mean(chroma)
#     sc = (np.mean((chroma - uc)**2))**0.5

#     #2nd term
#     top = np.int(np.round(0.01*l.shape[0]*l.shape[1]))
#     sl = np.sort(l,axis=None)
#     isl = sl[::-1]
#     conl = np.mean(isl[:top])-np.mean(sl[:top])

#     #3rd term
#     satur = []
#     chroma1 = chroma.flatten()
#     l1 = l.flatten()
#     for i in range(len(l1)):
#         if chroma1[i] == 0: satur.append(0)
#         elif l1[i] == 0: satur.append(0)
#         else: satur.append(chroma1[i] / l1[i])

#     us = np.mean(satur)

#     uciqe = c1 * sc + c2 * conl + c3 * us

#     return uciqe





def rgb2lab (inputColor) :
    
   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
    #    value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab