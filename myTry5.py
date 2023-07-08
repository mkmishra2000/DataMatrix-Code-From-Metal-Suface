###########################################################
# Name         :- Manas Kumar Mishra
# Organization :- IIITDM kancheepuram
# Project      :- DataMarrix code extraction from metal surface
# Guide        :- Dr. Rohini P
########################################################### 

import HomoMorphic as homo
import FeatureFusion as ff
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pylibdmtx.pylibdmtx import decode

def plotHistogram(img):
    """
    Plot histogram of given image
    input image
    output histogram of intensity
    """
    histr = cv2.calcHist([img], [0], None, [256],[0, 255])
    plt.plot(histr)
    plt.xlim([0, 256])
    plt.title('Histogram')
    plt.show()


def blockDefinition(img, rolpos, colpos, M, N):
    """
    This function is for divide the image into 
    blocks. 
    Input :- row wise position (rolpos), column wise position (colpos)
    Output:- Matrix of size M*N
    """

    for j in range(M):
        for i in range(N):
            Mat1[j][i] = img[(M*rolpos)+j][(N*colpos)+i]
    
    return Mat1


def claheEqualization(img, M, N):
    """
    Contrast limited adaptive histogram equalization 
    """
    clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(3,3))
    cl_img = clahe.apply(img)

    plt.imshow(cl_img, cmap='gray')
    plt.title('Clahe image')
    plt.show()

    return cl_img


def imageEnhanceing(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j]<64):
                img[i][j] = int(img[i][j]*0.75)
            elif(img[i][j]>63 and img[i][j]<128):
                img[i][j] = int(img[i][j]*0.9)
            elif(img[i][j]>127 and img[i][j]<192):
                img[i][j] = img[i][j]
            else:
                img[i][j] = int(img[i][j]*1.15)
                if img[i][j]>254:
                    img[i][j] = 255
    
    return img

def sumOfAllIntensity(Mat):
    sumOfele =0
    for i in range(Mat.shape[0]):
            for j in range(Mat.shape[1]):
                if i ==Mat.shape[0]//2 and j == Mat.shape[1]//2:
                    sumOfele += 2*Mat[i][j]
                elif i == 0 or i==Mat.shape[0]-1:
                    sumOfele += 0.5*Mat[i][j]
                elif j==0 or j==Mat.shape[1]-1 and i !=0 and i !=Mat.shape[0]-1:
                    sumOfele += 0.5*Mat[i][j]
                else:
                    sumOfele += Mat[i][j]

    return sumOfele

path = 'F:\\books\\5YearFinal_Project\\Internship\\images\\MT_QR8.jpg'

img = cv2.imread(path, 0)

maxValue = img.max()
print("Max value ", maxValue)

print(img.shape)

plt.imshow(img, cmap='gray')
plt.title('New image')
plt.show()

# img = cv2.GaussianBlur(img, (3,3),8, borderType=cv2.BORDER_REPLICATE)

row = img.shape[0]
col = img.shape[1]

extraRow = row%16
extraCol = col%16

print("Extra rows ", extraRow)
print("Extra columns ", extraCol)

if extraRow%2==0:
    uppercutRow = extraRow/2
    lowercutRow = extraRow/2
else:
    uppercutRow = (extraRow+1)/2
    lowercutRow = (extraRow-1)/2

if extraCol%2==0:
    uppercutCol = extraCol/2
    lowercutCol = extraCol/2
else:
    uppercutCol = (extraCol+1)/2
    lowercutCol = (extraCol-1)/2

for i in range(int(uppercutRow)):
    img = np.delete(img, 0, 0)

for i in range(int(lowercutRow)):
    img = np.delete(img, img.shape[0]-1, 0)

for i in range(int(uppercutCol)):
    img = np.delete(img, 0, 1)

for i in range(int(lowercutCol)):
    img = np.delete(img, img.shape[1]-1, 1)


print(img.shape)

plt.imshow(img, cmap='gray')
plt.title('Croped image')
plt.show()

# img = imageEnhanceing(img)


# plt.imshow(img, cmap='gray')
# plt.title('enhanced image')
# plt.show()

# img = np.multiply(img,2)
plotHistogram(img)


# equImg = cv2.equalizeHist(img)
# plotHistogram(equImg)
# plotHistogram(equImg)

newRow = img.shape[0]
newCol = img.shape[1]

M = int(newRow/16)
N = int(newCol/16)

print("Block rows ", M)
print("Block col  ", N)

cl_img = claheEqualization(img, M, N)

plotHistogram(cl_img)

# Converting image into binary image 
img_bin = cv2.threshold(cl_img, 110, 255, cv2.THRESH_BINARY)[1]

plt.imshow(img_bin, cmap='gray')
plt.title("binary image")
plt.show()


# Maximum = M*N*255
Maximum = 255*(M*N+1)-(18*255)

thershold = Maximum/2

print("Threshold ", thershold)

Mat1 = np.zeros((M, N))

for j in range(M):
    for i in range(N):
        Mat1[j][i] = img[(M*1)+j][(N*12)+i]

# print(Mat1.sum())
# plt.imshow(Mat1, cmap='gray')
# plt.title('New image')
# plt.show()

matrix = np.ones((16, 16))
matrix = np.multiply(matrix, 255)

rows = 16
columns = 16

for i in range(rows):
    for j in range(columns):
        if j == 0:
            matrix[i][j] = 0
        if i == 15:
            matrix[i][j] = 0
        if i ==0 and j%2 == 0:
            matrix[i][j] = 0
        if j==15 and i%2 != 0 and i != 0:
            matrix[i][j] = 0


print(matrix)

plt.imshow(matrix, cmap='gray')
plt.title("Basic pattern image")
plt.show()

rowWise = 1
colWise = 1


while(rowWise<15):
    colWise = 1
    while(colWise<15):
        Mat = blockDefinition(cl_img, rowWise, colWise, M, N)


        if(sumOfAllIntensity(Mat)<thershold):
            matrix[rowWise][colWise] = 255
        else:
            matrix[rowWise][colWise] = 0

        colWise += 1
    
    rowWise += 1


print("Print final")
print(matrix)

plt.imshow(matrix, cmap='gray')
plt.title("Final image")
plt.show()
