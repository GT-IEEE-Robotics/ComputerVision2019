import cv2
import numpy as np
import time
import glob

def colorMasks(inputImage):
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
    yellowMaskBlended = cv2.inRange(hsv, yellowLow, yellowHigh)

    ######BLUE RANGE#########
    #Yellow Mask
    blueLow = np.array([90,50,50])
    blueHigh = np.array([110,255,255])
    blueMask = cv2.inRange(hsv, blueLow, blueHigh)

    #Merged Mask
    rgbyMask = greenMask + redMask + yellowMaskBlended + blueMask
    blendedMask = cv2.bitwise_and(inputImage, inputImage, mask = rgbyMask)

    return blendedMask, redMask, blueMask, greenMask

def contourBoundWrite(colorMask):
    imageToCrop = colorMask
    grayscaledMask = cv2.cvtColor(imageToCrop, cv2.COLOR_BGR2GRAY)
    (contours, _) = cv2.findContours(grayscaledMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    generatedFilename = int(round(time.time() * 1000))
    for currentContour in contours:
        xPosition, yPosition, width, height = cv2.boundingRect(currentContour)
        
        if width > 60 and height > 60:
            generatedFilename += 1
            croppedImage = imageToCrop[yPosition: yPosition + height, xPosition: xPosition + width]
            cv2.imwrite(str(generatedFilename) + '.png', croppedImage)

def imageSegment():
    imageFiles = glob.glob("images/*.jpg")
    images = [cv2.imread(currentImage) for currentImage in imageFiles]

    for currentImage in images:
        
        currentScene = currentImage
        segmentedMask, redMask, blueMask, greenMask = colorMasks(currentScene) 
        segmentedImage = segmentedMask

        #RGBY Extraction
        blue = segmentedMask[:,:,0]
        green = segmentedMask[:,:,1]
        red = segmentedMask[:,:,2]
        
        blue = blue[np.where(blue > 170)]
        green = green[np.where(green > 170)]
        red = red[np.where(red > 170)]

        yellowLow = np.uint8([0,100,100])
        yellowHigh = np.uint8([100,255,255])
        yellow = cv2.inRange(segmentedMask, yellowLow, yellowHigh)
    
        greenMaskBlended = cv2.bitwise_and(currentScene, currentScene, mask = greenMask)
        blueMaskBlended = cv2.bitwise_and(currentScene, currentScene, mask = blueMask)
        redMaskBlended = cv2.bitwise_and(currentScene, currentScene, mask = redMask)
        yellowMaskBlended = cv2.bitwise_and(currentScene, currentScene, mask = yellow)
        
        contourBoundWrite(redMaskBlended)
        contourBoundWrite(greenMaskBlended)
        contourBoundWrite(blueMaskBlended)
        contourBoundWrite(yellowMaskBlended)
        
if __name__=="__main__":
    imageSegment()
