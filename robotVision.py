import cv2
import numpy as np
import time
import glob

def colorMasks(inputImage):
    hsv = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)

    redLow = np.array([0,120,70])
    redHigh = np.array([10,255,255])
    redLowMask = cv2.inRange(hsv, redLow, redHigh)

    redLow = np.array([170,120,70])
    redHigh = np.array([180,255,255])
    redHighMask = cv2.inRange(hsv, redLow, redHigh)

    redMask = redLowMask + redHighMask
    return redMask

def contourBoundWrite(colorMask, currentScene):
    
    grayscaledMask = cv2.cvtColor(currentScene, cv2.COLOR_BGR2GRAY)
    (contours, _) = cv2.findContours(grayscaledMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # generatedFilename = int(round(time.time() * 1000))
    for currentContour in contours:
        xPosition, yPosition, width, height = cv2.boundingRect(currentContour)
        
        if width > 150 and height > 25:
            # generatedFilename += 1
                
            cv2.rectangle(currentScene, (xPosition, yPosition), (xPosition+width, yPosition+height), (0, 255, 0), 8);
            
    currentScene = cv2.resize(currentScene, (1280, 720))
    cv2.imwrite('outputImage' + '.png', currentScene)
    # cv2.imshow('Corners', currentScene)
    # cv2.waitKey(0)

def imageSegment():
    imageFiles = glob.glob("images/*.jpg")
    images = [cv2.imread(currentImage) for currentImage in imageFiles]
    
    for currentImage in images:    
        currentScene = currentImage
        
        redMask = colorMasks(currentScene) 
        segmentedImage = redMask
        red = segmentedImage[:,:,0]
        red = red[np.where(red > 170)]
        
        redMaskBlended = cv2.bitwise_and(currentScene, currentScene, mask = redMask)
    
        contourBoundWrite(redMaskBlended, currentScene)
        
if __name__=="__main__":
    imageSegment()