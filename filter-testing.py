import os
import cv2
import numpy as np


# Gets the folder where the testing image is stored
imagesFolderPath = os.path.abspath(__name__).removesuffix("__main__")+"filter-testing-images\\"


# Creates the filter for the mass
lowerBoundsMass = []
upperBoundsMass = []
with open(os.path.abspath(__name__).removesuffix("__main__")+"mass-color-filter.txt") as file:
    lines = file.readlines()
    for i in range(3, 6):
        tempList = lines[i].split(" ")
        lowerBoundsMass.append(int(tempList[1]))
        upperBoundsMass.append(int(tempList[2]))

lowerBoundsMass = np.array(lowerBoundsMass)
upperBoundsMass = np.array(upperBoundsMass)


# Creates the filter for the mass
lowerBoundsRef = []
upperBoundsRef = []
with open(os.path.abspath(__name__).removesuffix("__main__")+"reference-color-filter.txt") as file:
    lines = file.readlines()
    for i in range(3, 6):
        tempList = lines[i].split(" ")
        lowerBoundsRef.append(int(tempList[1]))
        upperBoundsRef.append(int(tempList[2]))

lowerBoundsRef = np.array(lowerBoundsRef[::-1])
upperBoundsRef = np.array(upperBoundsRef[::-1])


# Iterates through all images in the folder
for fileName in os.listdir(imagesFolderPath):
    filePath = imagesFolderPath+fileName

    # Opens image
    video = cv2.VideoCapture(filePath)
    # Iterates through frames
    frameCountTrue = 0
    frameCount = 0
    timeData = []
    positionData = []
    while frameCount <= 30 * 45:
        # Gets frame
        ret, frame = video.read()

        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Breaks if no frame
        if not ret:
            break
        
        if frameCountTrue < 10*30:
            frameCountTrue += 1
            continue
        
        # Filters the image based on the color bounds for the mass
        filteredImage = cv2.inRange(frame2, lowerBoundsMass, upperBoundsMass)

        # Gets the average position
        yCoordinatesMass, xCoordinatesMass = np.where(filteredImage == 255)
        xAvgMass = sum(xCoordinatesMass)/len(xCoordinatesMass)
        yAvgMass = sum(yCoordinatesMass)/len(yCoordinatesMass)

        cv2.namedWindow("Unfiltered Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Unfiltered Image", cv2.resize(frame, (600, 300)))
        cv2.namedWindow("Filtered Image Mass", cv2.WINDOW_NORMAL)
        cv2.imshow("Filtered Image Mass", cv2.resize(filteredImage, (600, 300)))

        # Filters the image based on the color bounds for the reference
        filteredImage = cv2.inRange(frame, lowerBoundsRef, upperBoundsRef)
        
        cv2.namedWindow("Filtered Image Ref", cv2.WINDOW_NORMAL)
        cv2.imshow("Filtered Image Ref", cv2.resize(filteredImage, (600, 300)))

        # Gets the average position
        yCoordinatesRef, xCoordinatesRef = np.where(filteredImage == 255)
        xAvgRef = sum(xCoordinatesRef)/len(xCoordinatesRef)
        yAvgRef = sum(yCoordinatesRef)/len(yCoordinatesRef)

        # Calculates the angle
        angle = np.arctan2(xAvgMass-xAvgRef, yAvgMass-yAvgRef)
        
        print("Coordinate of mass:", xAvgMass, yAvgMass)
        print("Coordinate of reference:", xAvgRef, yAvgRef)
        print("Angle", np.arctan2(xAvgMass-xAvgRef, yAvgMass-yAvgRef))

        frameCount += 1
        cv2.waitKey(0)
        cv2.destroyAllWindows()