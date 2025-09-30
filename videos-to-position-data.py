import os
import cv2
import numpy as np


# Gets the folder where all the videos are stored
videoFolderPath = os.path.abspath(__name__).removesuffix("__main__")+"videos\\"

# Gets the folder where all the position data is stored
dataFolderPath = os.path.abspath(__name__).removesuffix("__main__")+"position-data\\"


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

# Gets the frame rate
frameRate = 60
with open(os.path.abspath(__name__).removesuffix("__main__")+"frame-rate.txt") as file:
    lines = file.readlines()
    frameRate = int(lines[2])


# Iterates through all files in videos folder
videoCount = 0
for fileName in os.listdir(videoFolderPath):
    filePath = videoFolderPath+fileName


    if fileName[:-10]+"-position-data.txt" in os.listdir(dataFolderPath):
        videoCount += 1
        print("Skipped:", videoCount)
        continue

    # Opens video
    video = cv2.VideoCapture(filePath)

    # Iterates through frames
    frameCountTrue = 0
    frameCount = 0
    timeData = []
    positionData = []
    uncertaintyData = []
    while frameCount <= frameRate * 45:
        # Gets frame
        ret, frame = video.read()

        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Breaks if no frame
        if not ret:
            break
        
        if frameCountTrue < 10*frameRate:
            frameCountTrue += 1
            continue
        
        # Filters the image based on the color bounds for the mass
        filteredImage = cv2.inRange(frameHSV, lowerBoundsMass, upperBoundsMass)

        # Gets the average position
        yCoordinatesMass, xCoordinatesMass = np.where(filteredImage == 255)
        if len(xCoordinatesMass) > 0:
            xAvgMass = sum(xCoordinatesMass)/len(xCoordinatesMass)
            yAvgMass = sum(yCoordinatesMass)/len(yCoordinatesMass)

            # Filters the image based on the color bounds for the reference
            filteredImage = cv2.inRange(frame, lowerBoundsRef, upperBoundsRef)

            # Gets the average position
            yCoordinatesRef, xCoordinatesRef = np.where(filteredImage == 255)
            if len(xCoordinatesRef) > 0:
                xAvgRef = sum(xCoordinatesRef)/len(xCoordinatesRef)
                yAvgRef = sum(yCoordinatesRef)/len(yCoordinatesRef)
                # Calculates the angle
                angle = np.arctan2(xAvgMass-xAvgRef, yAvgMass-yAvgRef)

                uncertainty = max(abs(np.arctan2(xAvgMass-xAvgRef+0.5, yAvgMass-yAvgRef+0.5)-angle), abs(np.arctan2(xAvgMass-xAvgRef+0.5, yAvgMass-yAvgRef-0.5)-angle), abs(np.arctan2(xAvgMass-xAvgRef-0.5, yAvgMass-yAvgRef-0.5)-angle), abs(np.arctan2(xAvgMass-xAvgRef-0.5, yAvgMass-yAvgRef+0.5)-angle))
            else:
                cv2.namedWindow("Unfiltered Image", cv2.WINDOW_NORMAL)
                cv2.imshow("Unfiltered Image", cv2.resize(frame, (600, 300)))
                cv2.namedWindow("Filtered Image Ref", cv2.WINDOW_NORMAL)
                cv2.imshow("Filtered Image Ref", cv2.resize(filteredImage, (600, 300)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                if len(positionData) == 0:
                    frameCountTrue += 1
                    continue
                else:
                    angle = positionData[-1]
                    uncertainty = uncertaintyData[-1]
        else:
            cv2.namedWindow("Unfiltered Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Unfiltered Image", cv2.resize(frame, (600, 300)))
            cv2.namedWindow("Filtered Image Mass", cv2.WINDOW_NORMAL)
            cv2.imshow("Filtered Image Mass", cv2.resize(filteredImage, (600, 300)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            if len(positionData) == 0:
                frameCountTrue += 1
                continue
            else:
                angle = positionData[-1]
                uncertainty = uncertaintyData[-1]
        
        # Adds position to data list
        positionData.append(angle)
        timeData.append(frameCount/frameRate)
        uncertaintyData.append(uncertainty)

        frameCount += 1
    
    positionData = positionData[:-3*frameRate]
    timeData = timeData[:-3*frameRate]
    uncertaintyData = uncertaintyData[:-3*frameRate]

    with open(dataFolderPath+fileName[:-10]+"-position-data.txt", "w") as file:
        for time, position, uncertainty in list(zip(timeData, positionData, uncertaintyData)):
            file.write(str(time) + " " + str(position) + " " + str(uncertainty))
            file.write("\n")
    
    videoCount += 1
    print(videoCount)