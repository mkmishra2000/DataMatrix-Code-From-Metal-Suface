###########################################################
# Name         :- Manas Kumar Mishra
# Organization :- IIITDM kancheepuram
# Project      :- DataMarrix code extraction from metal surface
# Guide        :- Dr. Rohini P
###########################################################

import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import io, color

# Plot the histogram of read image 
# [img] = read image, channel = [0], mask = [None], Histogram size =[256], range =[0, 256]
def plotHistogram(img):
    histr = cv2.calcHist([img], [0], None, [256],[0, 255])
    plt.plot(histr)
    plt.xlim([0, 256])
    plt.title('Histogram')
    plt.show()


# Find difference of Gaussian
def DiffOfGaussian(img, sigma1, sigma2):

    # Gaussian blur
    Img_blur1 = cv2.GaussianBlur(img, (7,7), sigma1, borderType=cv2.BORDER_REPLICATE)
    Img_blur2 = cv2.GaussianBlur(img, (5,5), sigma2, borderType=cv2.BORDER_REPLICATE)

    Img_blur = Img_blur1 - Img_blur2

    return Img_blur


# display the image till user pressed the any key 
def displayImg(img):

    # Show the image
    plt.imshow(img, cmap='gray')
    plt.title("Image under test")
    plt.show()

    print("Image has been closed...\n")


# K-means clustering (Binary segmentation) (Binary vector Quantization)
def kmeanClustering(image):
    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))

    # convert to float
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    k = 2
    compactness, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    # show the image
    # plt.imshow(segmented_image)
    # plt.show()

    # disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.copy(image)

    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))

    # put color black for all label 0
    cluster1 = 0
    cluster2 = 1

    if masked_image[labels==cluster1][0,0] >128:
        masked_image[labels == cluster1] = [255, 255, 255]
    # else:
        # masked_image[labels == cluster1]= [0, 0, 0]

    if masked_image[labels==cluster2][0,0]<=128:
        masked_image[labels == cluster2] = [0,0,0]
    # else:
    #     masked_image[labels == cluster2] = [255, 255, 255]

    
    # masked_image[labels == cluster1] = [255, 255, 255]
    # # print(masked_image[labels == cluster1])
    # masked_image[labels == cluster2] = [0, 0, 0]

    # Put white color for all label 1
    # masked_image[labels == cluster3] = [255, 255, 255]
    # masked_image[labels == cluster4] = [255, 255, 255]

    # convert back to original shape
    masked_image = masked_image.reshape(image.shape)

    # show the image
    plt.imshow(masked_image)
    plt.title("Masked image")
    plt.show()

    return masked_image


def MarphologicalOperation(img, Ksize):
    #kernal of Ksize*Ksize
    kernel1 = np.ones((Ksize,Ksize), np.uint8)

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1)

    displayImg(opening)

    kernel1 = np.ones((Ksize+1,Ksize+1), np.uint8)

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel1)

    return closing

if __name__ == '__main__':
    
    # Add path of the images
    path = 'F:\\books\\5YearFinal_Project\\Internship\\images\\QR8.jpg'

    # Read the image file on path
    img = cv2.imread(path)

    displayImg(img)

    plotHistogram(img)
    stdVal = img.std()

    print("RMS contrast of image under test ", stdVal)

    img = cv2.GaussianBlur(img, (3,3), 0.5, borderType=cv2.BORDER_REPLICATE)

    TextureImg = DiffOfGaussian(img, 2, 1)

    displayImg(TextureImg)

    segImg = kmeanClustering(img)

    # segImg = MarphologicalOperation(segImg)

    # Converting image into binary image 
    img_bin = cv2.threshold(segImg, 128, 255, cv2.THRESH_BINARY)[1]


    if(stdVal<=20):
        Ksize = 1
    elif(stdVal<=40 and stdVal>20):
        Ksize = 2
    elif(stdVal>40):
        Ksize = 3

    final_img = MarphologicalOperation(img_bin, Ksize)
    
    plt.imshow(final_img)
    plt.title("Final image")
    plt.show()


    
