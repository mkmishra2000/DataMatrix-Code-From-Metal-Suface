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


path = 'F:\\books\\5YearFinal_Project\\Internship\\images\\MT_QR8.jpg'

img = cv2.imread(path, 0)

maxValue = img.max()
print("Max value ", maxValue)

print(img.shape)

plt.imshow(img, cmap='gray')
plt.title('New image')
plt.show()

# img = cv2.GaussianBlur(img, (3,3),0.5, borderType=cv2.BORDER_REPLICATE)

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
plt.title('Cropped image')
plt.show()

plotHistogram(img)

equImg = cv2.equalizeHist(img)
plotHistogram(equImg)

newRow = img.shape[0]
newCol = img.shape[1]

M = int(newRow/16)
N = int(newCol/16)

print("Block rows ", M)
print("Block col  ", N)

Maximum = M*N*255

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

matrix = np.zeros((16, 16))

rows = 16
columns = 16

for i in range(rows):
    for j in range(columns):
        if j == 0:
            matrix[i][j] = 255
        if i == 15:
            matrix[i][j] = 255
        if i ==0 and j%2 == 0:
            matrix[i][j] = 255
        if j==15 and i%2 != 0 and i != 0:
            matrix[i][j] = 255

print(matrix)

plt.imshow(matrix, cmap='gray')
plt.title("Basic Pattern image")
plt.show()

rowWise = 1
colWise = 1


while(rowWise<15):
    colWise = 1
    while(colWise<15):
        Mat = blockDefinition(equImg, rowWise, colWise, M, N)

        if(Mat.sum()<thershold):
            matrix[rowWise][colWise] = 0
        else:
            matrix[rowWise][colWise] = 255

        colWise += 1
    
    rowWise += 1

print("Print final")
print(matrix)

plt.imshow(matrix, cmap='gray')
plt.title("Final image")
plt.show()

