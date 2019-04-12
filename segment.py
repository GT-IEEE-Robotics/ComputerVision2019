import cv2
import numpy as np
import time


def rgbySegment(inputImage):
    hsv = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)

    ######RED RANGE#########
    #Red Mask Lower
    redLow = np.array([0,120,70])
    redHigh = np.array([10,255,255])
    redLowMask = cv2.inRange(hsv, redLow, redHigh)

    #Red Mask Upper
    redLow = np.array([170,120,70])
    redHigh = np.array([180,255,255])
    redHighMask = cv2.inRange(hsv, redLow, redHigh)

    redMask = redLowMask + redHighMask

    ######GREEN RANGE#########
    #Green Mask
    greenLow = np.array([30,65,65])
    greenHigh = np.array([80,255,255])
    greenMask = cv2.inRange(hsv, greenLow, greenHigh)

    ######YELLOW RANGE#########
    #Yellow Mask
    yellowLow = np.array([20,20,20])
    yellowHigh = np.array([30,255,255])
    yellowMask = cv2.inRange(hsv, yellowLow, yellowHigh)

    ######BLUE RANGE#########
    #Yellow Mask
    blueLow = np.array([90,50,50])
    blueHigh = np.array([110,255,255])
    blueMask = cv2.inRange(hsv, blueLow, blueHigh)

    #Merged Mask
    rgbyMask = greenMask + redMask + yellowMask + blueMask
    blendedMask = cv2.bitwise_and(inputImage, inputImage, mask = rgbyMask)

    # outputImage = blendedMask
    # outputImageResized = cv2.resize(outputImage, (960,540))
    # cv2.imshow("Mask",outputImageResized)
    # cv2.waitKey(0)

    return blendedMask, redMask, blueMask, greenMask

def extractColor():

    extractionArray1 = np.zeros((4,))
    extractionArray2 = np.zeros((4,))

# while True:
    currentScene = cv2.imread('test.jpg')
    segmentedMask, redMask, blueMask, greenMask = rgbySegment(currentScene) #, redMask, blueMask, greenMask
    segmentedImage = segmentedMask

    blue = segmentedMask[:,:,0]
    green = segmentedMask[:,:,1]
    red = segmentedMask[:,:,2]

    #RGB Extraction
    blue = blue[np.where(blue > 170)]
    green = green[np.where(green > 170)]
    red = red[np.where(red > 170)]

    #Yellow Extraction
    yellowLow = np.uint8([0,100,100])
    yellowHigh = np.uint8([100,255,255])
    yellow = cv2.inRange(segmentedMask, yellowLow, yellowHigh)
    yellowNonZero1 = cv2.countNonZero(yellow)

    blackLow = np.uint8([0,0,0])
    blackHigh = np.uint8([0,0,0])
    black = cv2.inRange(segmentedMask, blackLow, blackHigh)
    blackNonZero = cv2.countNonZero(black)

    currentScene = cv2.imread('test.jpg')
    segmentedMask, redMask, blueMask, greenMask = rgbySegment(currentScene)
    segmentedImage = segmentedMask

    #RGB Extraction
    blueOne = segmentedMask[:,:,0]
    greenOne = segmentedMask[:,:,1]
    redOne = segmentedMask[:,:,2]

    blueOne = blueOne[np.where(blueOne > 170)]
    greenOne = greenOne[np.where(greenOne > 170)]
    redOne = redOne[np.where(redOne > 170)]

    # print(blueOne)
    # print(greenOne)
    # print(redOne)

    #Yellow Extraction
    yellowLow = np.uint8([0,100,100])
    yellowHigh = np.uint8([100,255,255])
    yellow = cv2.inRange(segmentedMask, yellowLow, yellowHigh)
    yellowNonZero2 = cv2.countNonZero(yellow)

    blackLow = np.uint8([0,0,0])
    blackHigh = np.uint8([0,0,0])
    black = cv2.inRange(segmentedMask, blackLow, blackHigh)
    black = cv2.countNonZero(black)

    extractionArray1 = np.array([red.size, green.size, blue.size, yellowNonZero1])
    extractionArray2 = np.array([redOne.size, greenOne.size, blueOne.size, yellowNonZero2])

    maximumValues1 = np.argmax(extractionArray1)
    maximumValues2 = np.argmax(extractionArray2)

    blackThreshold = 0.96

    if black > (blackThreshold * len(segmentedMask) * len(segmentedMask[0])):
        maximumValues2 = 4

    if blackNonZero > (blackThreshold * len(segmentedImage) * len(segmentedImage[0])):
        maximumValues1 = 5


    finalImage = str(maximumValues1) + "|" + str(maximumValues2)
    finalArray = np.array([maximumValues1, maximumValues2])

    outputImage = segmentedImage
    outputImageResized = cv2.resize(outputImage, (960,540))

    # cv2.imshow('Blue', blueMask)
    # cv2.imshow("Green", greenMask)
    # cv2.imshow("Red", redMask)
    # cv2.imwrite("current.jpg", currentScene)
    # cv2.imwrite("image2.jpg", outputImageResized)
    greenMasked = cv2.bitwise_and(currentScene, currentScene, mask = greenMask)
    # cv2.imwrite('green.jpg', final)
    blueMasked = cv2.bitwise_and(currentScene, currentScene, mask = blueMask)
    # cv2.imwrite('blue.jpg', final)
    redMasked = cv2.bitwise_and(currentScene, currentScene, mask = redMask)
    # cv2.imwrite('red.jpg', final)
    yellowMasked = cv2.bitwise_and(currentScene, currentScene, mask = yellow)
    # cv2.imwrite('yellow.jpg', final)
    
    # test = redMasked
    # test = cv2.resize(test, (600, 600))
    # cv2.imshow('test',test)
    # cv2.waitKey(0)
    
    imageToCrop = redMasked
    grayscaledMask = cv2.cvtColor(imageToCrop, cv2.COLOR_BGR2GRAY)
    detectedBoundaries = cv2.Canny(imageToCrop, 100, 150)
    (contours, _) = cv2.findContours(detectedBoundaries.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    index = 0
    for current in contours:
        x,y,w,h = cv2.boundingRect(current)
        if w > 50 and h > 50:
            index += 1
            croppedImage = imageToCrop[y :y+h, x:x+w]
            cv2.imwrite(str(index) + '.png', croppedImage)
    print('redcropped')
    
    # imageToCrop = greenMasked
    # grayscaledMask = cv2.cvtColor(imageToCrop, cv2.COLOR_BGR2GRAY)
    # detectedBoundaries = cv2.Canny(imageToCrop, 10, 250)
    # (contours, _) = cv2.findContours(detectedBoundaries.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 
    # index = 0
    # for current in contours:
    #     x,y,w,h = cv2.boundingRect(current)
    #     if w > 50 and h > 50:
    #         index += 1
    #         croppedImage = imageToCrop[y :y+h, x:x+w]
    #         cv2.imwrite(str(index) + '.png', croppedImage)
    # print('greencropped')
    # 
    # imageToCrop = blueMasked
    # grayscaledMask = cv2.cvtColor(imageToCrop, cv2.COLOR_BGR2GRAY)
    # detectedBoundaries = cv2.Canny(imageToCrop, 10, 250)
    # (contours, _) = cv2.findContours(detectedBoundaries.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 
    # index = 0
    # for current in contours:
    #     x,y,w,h = cv2.boundingRect(current)
    #     if w > 50 and h > 50:
    #         index += 1
    #         croppedImage = imageToCrop[y :y+h, x:x+w]
    #         cv2.imwrite(str(index) + '.png', croppedImage)
    # 
    # print('bluecropped')
            
    # imageToCrop = yellowMasked
    # grayscaledMask = cv2.cvtColor(imageToCrop, cv2.COLOR_BGR2GRAY)
    # detectedBoundaries = cv2.Canny(imageToCrop, 10, 250)
    # (contours, _) = cv2.findContours(detectedBoundaries.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 
    # index = 0
    # for current in contours:
    #     x,y,w,h = cv2.boundingRect(current)
    #     if w > 50 and h > 50:
    #         index += 1
    #         croppedImage = imageToCrop[y :y+h, x:x+w]
    #         cv2.imwrite(str(index) + '.png', croppedImage)        
    # print('yellowcropped')
    # 
    # cv2.imshow("im",image)
    
    # cv2.waitKey(0)
    # wierd = cv2.Canny(final,100,200)
    # contours = cv2.findContours(wierd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # rect = cv2.boundingRect(contours[0])
    # crop_img = final[rect[0]:rect[3], rect[1]:rect[2]]
    # cv2.imwrite("hi.jpg", crop_img)

if __name__=="__main__":
    extractColor()
