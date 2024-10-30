# Program designed to identify and predict the total number of "bug splats" on a provided number plate.
# 3rd Year project artifact of George Compton.


#Import neccessary modules/files
import numpy
import cv2 #computer vision library
import csv
import os #filepath
import random #testing

#Define functions used multiple times
#AreInSeries used to check if objects on plate exist (or are very close) at same dimension along given axis at any point (first step in deciding if in series)
def AreInSeries (StartingPos1, Length1, StartingPos2, Length2, Threshold):
    for edge1 in range(StartingPos1, StartingPos1+Length1+1):
        for edge2 in range(StartingPos2, StartingPos2+Length2+1):
            if abs(edge1-edge2) < Threshold:
                return True
    else:
        return False

#IsAtEdge used to check if object exists at edge of plate along a given axis at any point
def IsAtEdge (ImageLength, ObjectStartingPos, ObjectLength, Threshold):
    if (ObjectStartingPos <= ImageLength*Threshold) or (ObjectStartingPos+ObjectLength >= ImageLength*(1-Threshold)):
        return True
    else:
        return False

#MAIN CODE BEGINS HERE
#Import initial data, so range is established for plate selection
DataLocation = os.path.join(os.path.dirname(__file__), "Input Data") #Get filepath for input data
InputData = [] #Create empty list
for i in os.listdir(DataLocation): #For image in input data folder, append image to empty list
    InputData.append(cv2.imread(DataLocation+"\\"+i))

File = open("Plate Type.csv") #csv simple, easy to flatten to 1d array or add to if more could/should be tracked
IsFrontPlate = list(csv.reader(File, delimiter=","))
IsFrontPlate = numpy.array(IsFrontPlate).flatten()

#Introductory screen/instructions on use
print("This program will attempt to get the number of bug splats on the selected image/images from the data provided. For more information, please read the 'Instructions.txt' file within the program. ")
while True:
    Input1 = input("please enter first number: ")
    Input2 = input("please enter second number: ")
    try: 
        Input1, Input2 = int(Input1), int(Input2)
    except ValueError: #do again if not whole numbers
        print("there has been an error. Please try again, and remember to enter two whole numbers. ")
        continue
    if (Input1 <= 0) or (Input2 > len(InputData)) or (Input1 > Input2): #do again if outside of range
        print("there has been an error. Please try again, and remember to enter the smaller number first, without entering numbers out of bounds (for example, negative one or nine quadrillion). ")
        continue
    break
InputData = InputData[Input1-1:Input2]
#print(len(InputData)) #for testing

#Prepare testing
TestImage = (random.randint(1, len(InputData)))-1
print("test image is labelled plate", TestImage+Input1, "in the dataset. ")

#Remove coloured "tags" sections from numberplate, before preprocessing/resizing 
UntaggedData = []
LowerYellow, UpperYellow = numpy.array([15, 100, 100], dtype = "uint8"), numpy.array([50, 255, 255], dtype = "uint8") #declare acceptable colour ranges
LowerWhite, UpperWhite = numpy.array([0, 0, 100], dtype = "uint8"), numpy.array([180, 50, 255], dtype = "uint8")
LowerBlack, UpperBlack = numpy.array([0, 0, 0], dtype = "uint8"), numpy.array([180, 255, 100], dtype = "uint8")

for i in range(len(InputData)):
    HSVImage = cv2.cvtColor(InputData[i], cv2.COLOR_BGR2HSV)
    PlateFeatures = cv2.inRange(HSVImage, LowerBlack, UpperBlack)
    if IsFrontPlate[i] == "True": #quotes as reading in text from csv
        Background = cv2.inRange(HSVImage, LowerWhite, UpperWhite)

    else:
        Background = cv2.inRange(HSVImage, LowerYellow, UpperYellow)

    BackMask = cv2.bitwise_not(Background)
    FeatureMask = cv2.bitwise_not(PlateFeatures)
    TagMask = cv2.bitwise_and(FeatureMask, BackMask)

    #emphasise pixels that get through mask
    BGRImage = cv2.cvtColor(HSVImage, cv2.COLOR_HSV2BGR)
    TagMask = cv2.cvtColor(TagMask, cv2.COLOR_GRAY2BGR)
    ContrastImage = cv2.addWeighted(BGRImage, 0.5, TagMask, 0.5, 0)
    UntaggedData.append(ContrastImage)
#cv2.imshow("untagged image", UntaggedData[TestImage]), cv2.waitKey(0) #for testing

#Preprocess images to enhance results
ProcessedData = []
for i in UntaggedData: #For image in data set, binarize at input threshold and add to binarized data set
    i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) #Get Greyscale
    
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(1,1)) #Somewhat normalise contrast/mitigate reflectiveness
    i = clahe.apply(i)

    ChangeToReach = (800000/(i.shape[0]*i.shape[1]))**0.5 #Resize to get constant area while keeping aspect ratio the same
    NewSize = (int(i.shape[1]*ChangeToReach), int(i.shape[0]*ChangeToReach))
    i = cv2.resize(i, NewSize, interpolation=cv2.INTER_LINEAR) #Uses linear interpolation

    i = cv2.fastNlMeansDenoising(i, h=2) #Do noise reduction to reduce non-bug splat plate inconsistencies

    ProcessedData.append(i)
#cv2.imshow("preprocessed image", ProcessedData[TestImage]), cv2.waitKey(0) #for testing

#Isolate plate background for analysis
#Binarize images
BinaryData = []
for i in ProcessedData: #For image in Data data set, binarize at threshold or blocksize/constant, and add to binarized data set
    BinaryData.append(cv2.adaptiveThreshold(i, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 10)) #Small blocksize to prioritize splats over whole letters
#cv2.imshow("binary image", BinaryData[TestImage]), cv2.waitKey(0) #for testing

#Remove objects that aren't bug splats
BugsplatData = []
BugCount, RemovedArea = [0] * len(BinaryData), [0] * len(BinaryData)
for i in range(len(BinaryData)):
    Keepers = numpy.zeros_like(BinaryData[i]) #Blank image to slowly add objects that we'll keep onto
    ObjectCount, Objects, Values, Centroids = cv2.connectedComponentsWithStats(255-BinaryData[i]) #255-i is to invert as white = component
    ViableObject = [True] * ObjectCount #assume all objects will be kept until proven otherwise
    SumOfCovered, PotentialRemovedArea = [0] * ObjectCount, [0] * ObjectCount
    
    for j in range(ObjectCount): 

        Area = Values[j, cv2.CC_STAT_AREA]
        if not (Area <= 1000): #Remove objects not bugsplat size (e.g large letters)
            ViableObject[j] = False
            SumOfCovered[j] = 800000 #image size
            PotentialRemovedArea[j] = Area
            continue
        elif not ((Area >= 10) and (Area <= 250)): #Want to remove actual plate letters without checking if in series, but still do so for other large objects
            ViableObject[j] = False
            PotentialRemovedArea[j] = Area

        CentroidsInSeries = [] #Initialise/Reinitialise variables
        Width = Values[j, cv2.CC_STAT_WIDTH]
        Height = Values[j, cv2.CC_STAT_HEIGHT]
        StartingX = Values[j, cv2.CC_STAT_LEFT]
        StartingY = Values[j, cv2.CC_STAT_TOP]
        
        if IsAtEdge(BinaryData[i].shape[1], StartingX, Width, 0.01) or IsAtEdge(BinaryData[i].shape[0], StartingY, Height, 0.01) == True: #Remove all objects at very edge of image
            ViableObject[j] = False
            SumOfCovered[j] = 800000 #image size
            PotentialRemovedArea[j] = Area
            continue
        
        if SumOfCovered[j] == 0:
            SumOfCovered[j] = Height*Width #Derive variables
        AtXEdge = [IsAtEdge(BinaryData[i].shape[1], StartingX, Width, 0.05), False]
        AtYEdge = [IsAtEdge(BinaryData[i].shape[0], StartingY, Height, 0.05), False]

        for k in range(j+1, ObjectCount): #Remove objects in perfect series (e.g smaller text)
            Width2 = Values[k, cv2.CC_STAT_WIDTH] #Initialise/Reinitialise variables
            Height2 = Values[k, cv2.CC_STAT_HEIGHT]
            StartingX2 = Values[k, cv2.CC_STAT_LEFT]
            StartingY2 = Values[k, cv2.CC_STAT_TOP]

            InSeriesX = AreInSeries(StartingX, Width, StartingX2, Width2, 10) #Derive variables
            InSeriesY = AreInSeries(StartingY, Height, StartingY2, Height2, 5)
            AtXEdge[1] = IsAtEdge(BinaryData[i].shape[1], StartingX2, Width2, 0.05)
            AtYEdge[1] = IsAtEdge(BinaryData[i].shape[0], StartingY2, Height2, 0.05)
            
            #if in series
            if (InSeriesX and InSeriesY == True) and (AtXEdge or AtYEdge == [True, True]): #If objects close enough in both dimensions and both at edge(ish) of image
                if len(CentroidsInSeries) == 0:
                    CentroidsInSeries.extend([j,k])
                else:
                    CentroidsInSeries.append(k)
                SumOfCovered[j] = SumOfCovered[j] + (Height2*Width2)

        if len(CentroidsInSeries) !=0:
            for l in range(len(CentroidsInSeries)):
                    SumOfCovered[CentroidsInSeries[l]] = SumOfCovered[j]
                    ViableObject[CentroidsInSeries[l]] = False
                    PotentialRemovedArea[CentroidsInSeries[l]] = Values[CentroidsInSeries[l], cv2.CC_STAT_AREA]


    #Determine if smaller potential bugsplats are just small plate features 
    for j in range(ObjectCount):
        if (ViableObject[j] == True) or (SumOfCovered[j] < 200):
            Mask = (Objects==j).astype("uint8")*255 #Fill whole image that isn't object w/ area j + make same image type
            Keepers = cv2.bitwise_or(Keepers, Mask) #Add object to output image
            BugCount[i] = BugCount[i] + 1
        else:
            RemovedArea[i] = RemovedArea[i]+PotentialRemovedArea[j]
    RemovedArea[i] = RemovedArea[i] - PotentialRemovedArea[0] #remove background

    BugsplatData.append(255-Keepers) #Append to final data set

#Get total percentage coverage and output results
for i in range(len(BugsplatData)):
    TotalCoverageRatio = 800000/(800000-RemovedArea[i])
    print(str(BugCount[i]) + " discrete bugsplats have been counted in image " + str(i+1) + ".")
    print("Adjusting for areas where bugsplats may have gone undetected, an estimated total count is "+ str(round(BugCount[i]*TotalCoverageRatio)) + " bugsplats.\n")

#dedicated testing section
#cv2.imshow("bugsplat image", BugsplatData[TestImage]), cv2.waitKey(0)
#for i in range(len(BugsplatData)):
    #cv2.imshow("bugsplat images", BugsplatData[i]), cv2.waitKey(0)



