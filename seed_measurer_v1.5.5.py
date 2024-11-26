#!/bin/python3

# Shebang that points to your python interpreter
# This should be changed based on your python environent
# and must remain on the program's first line

"""
 Benjamin Sims
 benjamis20@berkeley.edu

 Blackman Lab @ UC Berkeley
 Last Updated July 11, 2024

 Seed Measurer takes an image of seeds on a bright background and measures each seed's
 area, length, and width to a CSV file. Your seeds must be pictured inside a box of known
 width. The program will attempt to correct for perspecitve skew based on the shape of your box.

 Image file names must be in the format: [populationName]_[*]_[*]_[year]_[*].[imageFileExtension]

 Example CSV output for filename "NUB_8_7_2016_3_X_7_6_2016_4.jpg":

 genotype                        population   year    seeds_imaged	area_mm2	length_mm	width_mm
 NUB_8_7_2016_3_X_7_6_2016_4     NUB          2016    222           0.1982	    0.6711	    0.3968
 NUB_8_7_2016_3_X_7_6_2016_4     NUB          2016    222           0.1968	    0.6812	    0.3600
 NUB_8_7_2016_3_X_7_6_2016_4     NUB          2016    222           0.1911	    0.6520	    0.3575
 NUB_8_7_2016_3_X_7_6_2016_4     NUB          2016    222           0.1907	    0.6730	    0.3582
"""

# Import necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from time import time
from math import log10
import numpy as np
import argparse
import imutils
import threading
import csv
import cv2
import os

# Finds the midpoint between two sets of (x, y) coordinates
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Finds seeds in an individual photo
class PhotoReader:
    def __init__(self, inDir, outDir, imgName, boxWidth=100, marginPercent=5, seedThreadCount=8, seedRatio=3):

        # Define variables
        self.seedCount = 0
        self.outDir = outDir
        self.imgName = imgName
        self.boxWidth = boxWidth
        self.seedRatio = seedRatio
        self.imgOriginal = cv2.imread(os.path.join(inDir, imgName))
        self.imgCropped = None
        self.boxPts = None
        self.boxContour = None
        self.seedContours = []
        self.seedMeasurements = []
        self.pxPerMil = 0
        self.xBound = 0
        self.yBound = 0
        self.margin = 0
        self.marginPercent = marginPercent / 100
        self.seedThreadCount = seedThreadCount # Number of threads to use to find seeds in a list of potetial seed contours
        self.threads = []
        self.lock = threading.Lock()

        # Define CSV headers, then
        # get line name from the image filename
        self.headers = ["genotype", "population", "year", "seeds_imaged", "area_mm2", "length_mm", "width_mm"]
        lineList = imgName.split("/")[-1].split(".")[:-1]
        self.lineName = ".".join(lineList)

        # Get population and year from file name
        infoList = self.lineName.split("_")
        self.population = "NA"
        self.year = "NA"
        if len(infoList) > 3:
            self.population = infoList[0].upper()
            self.year = infoList[3]
        else:
            self.lock.acquire()
            print("WARNING: Unable to get population name and/or year from {0}. Recording values as NA".format(imgName))
            self.lock.release()

        # Perform image processing to find seeds
        # when a new PhotoReader object in defined
        self.findBox()
        self.cropImage()
        self.findSeedContours()
        self.setPixelMetric()
        self.findSeeds()

    # Finds bounding box in image
    def findBox(self):

        # Covert to grayscale
        gray = cv2.cvtColor(self.imgOriginal, cv2.COLOR_BGR2GRAY)

        # Otsu's thresholding to find dark and light areas of the image
        thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        # Uncomment this line if the program struggles to find box contour
        #thresh = cv2.erode(thresh, None, iterations=1)

        # find contours in the edge map
        # ignore the first contour that usually corresponds to the whole image
        cnts = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
     
        # Sort countours by area
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[1:]

         # The largest contour is the bounding box
        boxContour = cnts[0]

        epsilon = 0.02 * cv2.arcLength(boxContour, True)
        approx = cv2.approxPolyDP(boxContour, epsilon, True)

        # Check if the approximated contour has four points (a quadrilateral)
        if len(approx) == 4:
            self.boxPts = approx.reshape(4, 2)
        # Raise an error if no box is found
        else:
            raise Exception("Unable to find bounding box")

    # Crops an image looking for a large quadrilateral, corrects for distortion
    # and returns both the corrected black-and-white threshholded along with the corrected color original
    def cropImage(self):

        # Order points of bounding box in the order: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        s = self.boxPts.sum(axis=1)
        rect[0] = self.boxPts[np.argmin(s)]
        rect[2] = self.boxPts[np.argmax(s)]

        diff = np.diff(self.boxPts, axis=1)
        rect[1] = self.boxPts[np.argmin(diff)]
        rect[3] = self.boxPts[np.argmax(diff)]

        (tl, tr, br, bl) = rect

        # Calculate the width and height of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Get the perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)

        # Warp the input image with the computed transform matrix
        # to correct for any perspective skewing
        self.imgCropped = cv2.warpPerspective(self.imgOriginal, M, (maxWidth, maxHeight))

    # Finds all the seeds in the cropped image
    def findSeedContours(self):

        # Covert cropped image to grayscale
        gray = cv2.cvtColor(self.imgCropped, cv2.COLOR_BGR2GRAY)

        """
        *** IMPORTANT ***

        The line below is resonsable for adaptive thresholding to help find seeds in images with a wide
        variety of exposures.

        The second to last parameter of the below line represents the size of the box to sample
        when determining the image brightness. Keep this sed to 199 in higher resolution images.

        The last parameter is the sensitivity of the of the threshholding. Higher values are less sensitive,
        so only the darkest seeds will show up. Lower values are more sensitive which may result in erroneous
        seed detection. You might have to play around with this value if the program is consistently missing
        seeds or detecting seeds where none are present.
        """

        warpedThresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,199,50)

        # Find contours from the threshold image
        cnts = cv2.findContours(warpedThresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]

        # Sort countours by area
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

        # The largest contour is the bounding box
        self.boxContour = cnts[0]

        # The rest of the contours are seeds
        self.seedContours = cnts[1:]

        # Save margin metrics
        self.margin = int(self.marginPercent * warpedThresh.shape[1])
        self.xBound = warpedThresh.shape[1] - self.margin
        self.yBound = warpedThresh.shape[0] - self.margin

    def setPixelMetric(self):
        # compute the rotated bounding box of the contour
        box = cv2.minAreaRect(self.boxContour)
        box = cv2.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order
        box = perspective.order_points(box)
                
        # unpack the ordered bounding box, then
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tl, tr, br, bl) = box
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # compute the Euclidean distance between the midpoints
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # set the pixel per metric
        self.pxPerMil = dB / self.boxWidth

    # Finds the seeds in an image
    # and saves the contours of all the seeds
    def findSeeds(self):

        self.seedCount = len(self.seedContours)

        # Initialize array to hold seed contours
        filteredContours = []

        # Split seed contours into group[s for ultithreading support
        imgPerThread = self.seedCount // self.seedThreadCount

        # If there are fewer images than threads, set number of threads to number of images
        if not imgPerThread:
            self.seedThreadCount = self.seedCount

        # Split list of images into chunks for each thread
        chunks = []
        for i in range(self.seedThreadCount):
            chunks.append([])
        for i, contour in enumerate(self.seedContours):
            chunks[i%self.seedThreadCount].append(contour)

        # Local function that finds seeds in a chunk of contours
        def findSeed(contourChunk):
            # Discard seeds that are too close to the margin,
            # too small, or too thin
            for contour in contourChunk:
                x, y, w, h = cv2.boundingRect(contour)
                if x > self.margin and y > self.margin and (x + w) < self.xBound and (y + h) < self.yBound:

                    # To get area, length, and width from contour:
                    # first, compute the rotated bounding box of the contour
                    box = cv2.minAreaRect(contour)
                    box = cv2.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                    box = np.array(box, dtype="int")

                    # order the points in the contour such that they appear
                    # in top-left, top-right, bottom-right, and bottom-left order
                    box = perspective.order_points(box)
                            
                    # unpack the ordered bounding box, then compute the midpoint
                    # between the top-left and top-right coordinates, followed by
                    # the midpoint between bottom-left and bottom-right coordinates
                    (tl, tr, br, bl) = box
                    (tltrX, tltrY) = midpoint(tl, tr)
                    (blbrX, blbrY) = midpoint(bl, br)

                    # compute the midpoint between the top-left and top-right points,
                    # followed by the midpoint between the top-righ and bottom-right
                    (tlblX, tlblY) = midpoint(tl, bl)
                    (trbrX, trbrY) = midpoint(tr, br)

                    # compute the Euclidean distance between the midpoints
                    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                    # compute the area, length, and width in mm
                    area = cv2.contourArea(contour) / self.pxPerMil ** 2
                    metrics = [dA / self.pxPerMil, dB / self.pxPerMil]
                    width, length = min(metrics), max(metrics)

                    # if the length, width, or area is 0, skip
                    if (not length) or (not width) or (not area):
                        continue
                    # If the contour is not too thin to be a seed, save it
                    if length / width < self.seedRatio:
                        # Make sure to synchronize when writing to variables
                        self.lock.acquire()
                        self.seedMeasurements.append([area, length, width])
                        # Add contour to global filteredContour array
                        filteredContours.append(contour)
                        self.lock.release()

        # Initializes each thread for each subset of contours
        for i in range(self.seedThreadCount):
            currentThread = threading.Thread(target=findSeed, args=(chunks[i],))
            currentThread.start()
            self.threads.append(currentThread)

        # Ends each thread
        for thread in self.threads:
            thread.join()
        
        # Redifine seedContours class variable
        self.seedContours = filteredContours
    
    # Filter seeds based on deviation from the mean
    def filterSeeds(self, stdevs=3):

        # Find area, length, width of a seed 3 SDs less and more than the mean
        metrics = []
        for i in range(3):
            arr = [dataArr[i] for dataArr in self.seedMeasurements]
            std = np.std(arr)
            mean = np.mean(arr)
            metrics.append([mean - stdevs * std, mean + stdevs * std])

        # Filter out seeds that are too small or too big
        seedMeasurements = []
        filteredContours = []
        for seed, contour in zip(self.seedMeasurements, self.seedContours):
            record = True
            for i in range(3):
                if not (metrics[i][0] < seed[i] < metrics[i][1]):
                    record = False
                    break
            if record:
                seedMeasurements.append(seed)
                filteredContours.append(contour)
            
        self.seedMeasurements, self.seedContours = seedMeasurements, filteredContours 
        self.seedCount = len(seedMeasurements)

        return self.seedCount

    # Draws found seed contours to on warped origninal image,
    # saves image as jpeg
    def drawContours(self, downsizeRatio=1, color="red"):

        # Sets the color the seed contours will appear
        # in the output image
        colorTuple = (0,0,255)
        if color.lower() == "blue":
            colorTuple = (255,0,0)
        elif color.lower() == "green":
            colorTuple = (0,255,0)

        # Draw seed contours over corrected and cropped original image
        cv2.drawContours(self.imgCropped, self.seedContours,\
            -1,colorTuple,thickness=cv2.FILLED)

        # Downsize the image if a proper ratio is provided
        if 0 < downsizeRatio < 1:
             self.imgCropped = cv2.resize(self.imgCropped,
                        (0,0), # set fx and fy, not the final size
                        fx=downsizeRatio, 
                        fy=downsizeRatio, 
                        interpolation=cv2.INTER_NEAREST)
        
        # Write to out directory
        cv2.imwrite(os.path.join(self.outDir, (self.lineName + "_seeds.jpg")), self.imgCropped)

    # Writes data to CSV file
    def writeCSV(self):

        # Create a csv file
        with open(os.path.join(self.outDir, (self.lineName + ".csv")), "w", newline="") as csvFile:

            # Initialize csv writer
            seedWriter = csv.writer(csvFile)

            # Write header row
            seedWriter.writerow(self.headers)

            # loop over the seeds individually
            for seed in self.seedMeasurements:

                # Initialize the row to write to a csv
                # These information in these coloums will be the same in all rows of output
                row = [self.lineName, self.population, self.year, self.seedCount]
                
                # Labmda function to shorten measurment ro 4 decimals
                fourDecimals = lambda num : format(num, ".4f")

                # Write the area, length, and width to the row
                for metric in seed:
                    row.append(fourDecimals(metric))

                # Write the seed's dimenstions to the CSV
                seedWriter.writerow(row)

class PhotoManager:
    def __init__(self, progVersion, dateModified):
        self.progName = "Seed Measurer"
        self.ver = progVersion
        self.author = "Benjamin Sims"
        self.dateModified = dateModified

        self.startTime = time()

        # Creates an argument parser to get input from the command line
        ap = argparse.ArgumentParser(prog=self.progName,
        description="Seed Measurer takes an image of seeds on a bright background and measures their area, length, and width to a CSV file.\
            Your seeds must be pictured inside a box of known width. The program will attempt to correct for perspecitve skew based on the shape of your box.")

        ap.add_argument("-i", "--indir", required=True,
            help="Path to an individual image or directory of images containing your seeds")
        ap.add_argument("-m", "--margin", required=False, default=5,
            help="Anything within this percent margin inside the box will be ignored. Setting this value too low may cause issues (default: 5)")
        ap.add_argument("-o", "--outdir", required=False,
            help="Processed image output directory. Defaults to your home directory if no value is supplied")
        ap.add_argument("-t", "--threads", required=False, default=8,
            help="The number of threads used to process images (default: 8)")
        ap.add_argument("-v", "--version", required=False, nargs="?", const=True,
                        help="Show version info and exit")
        ap.add_argument("-w", "--width", required=False,
            help="Width of the box around your seeds in millimeters")
        args = vars(ap.parse_args())

        # Displays credits, quit if flag specified
        if args["version"]:
            self.version()

        self.threadCount = abs(int(args["threads"]))
        self.boxWidth = float(args["width"])
        self.outDir = args["outdir"]
        self.inDir = args["indir"]
        self.marginPercent = float(args["margin"])

        # Displays error if no width or zero width supplied
        if not self.boxWidth:
            print("Seed Measurer: error: the following argument is required and should be non-zero: -w/--width")
            exit(1)

        self.fileFormats = ["jpeg", "jpg", "tiff", "gif", "png", "heic"]
        self.dirList = [] # Records a list of image file paths

        # If an indir is a directory, read files that are in a supported file format
        tempList = []
        if os.path.isdir(self.inDir):
            tempList = os.listdir(self.inDir)

        # If indir is a file, add it to the list
        else:
            tempList.append(self.inDir)
        for file in tempList:
            if file.split(".")[-1].lower() in self.fileFormats and file[0] != ".":
                self.dirList.append(file)

        # If no images were found in the provided directory, raise an error
        if not self.dirList:
            raise Exception("No images found in provided directory " + self.inDir)
        
        # Record number of photos to be analyzed
        self.photoCount = len(self.dirList)
        self.photosSeen = 0
        self.errors = 0 # Number of images that failed to analyze

        # Saves the number of digits of the image count
        # to help format the terminal output
        self.digits = str(int(log10(self.photoCount)) + 1)

        # Multithreading support

        # Define array to hold threads
        # and lock to prevent a race condition
        self.threads = []
        self.lock = threading.Lock()

    # Prints credit info and quits
    def version(self):
        print(self.progName, self.ver, "by", self.author, self.dateModified)
        exit(0)

    # Function to execute on each thread
    def execute(self, images, seedThreadCount):
        # Loop over all image files
        for file in images:

            # Another photo is found
            self.lock.acquire()
            self.photosSeen += 1
            self.lock.release()

            # Assume that no seeds were found
            seedCount = 0

            # Print image processing progress
            self.lock.acquire()
            print(("\n({1:0" + self.digits + "d}/{2:0" + self.digits + "d}) Reading from {0}").format(file, self.photosSeen, self.photoCount))
            self.lock.release()
            # Try to analyze the image
            # Alert user if no seeds are found
            try:
                # Define a new PhotoReader object, passing it necessary parameters including the file path
                # to the image being analyzed
                seedReader = PhotoReader(self.inDir, self.outDir, file, self.boxWidth, self.marginPercent, seedThreadCount)
                # Filter out seeds that are too small or too big
                seedCount = seedReader.filterSeeds()
                # Save seed contour visiualization to file
                seedReader.drawContours()
                # Write data to CSV file
                seedReader.writeCSV()
                if not seedCount:
                    self.lock.acquire()
                    print("ERROR in {0}: No seeds found in image".format(file))
                    self.errors += 1
                    self.lock.release()
            
            # If an error is encountered while analyzing an image
            # alert user and move on to the next image
            except Exception as inst:
                self.lock.acquire()
                print("ERROR in {0}:".format(file), end = " ")
                print(inst)
                self.errors += 1
                self.lock.release()

    # Allocates images per thread, begins threaded image analysis
    # also allocates thread to find seeds within an image
    def begin(self, seedThreadCount=8):
        imgPerThread = self.photoCount // self.threadCount

        # If there are fewer images than threads, set number of threads to number of images
        if not imgPerThread:
            self.threadCount = self.photoCount

        # Split list of images into chunks for each thread
        chunks = []
        for i in range(self.threadCount):
            chunks.append([])
        for i, file in enumerate(self.dirList):
            chunks[i%self.threadCount].append(file)

        # Initializes each thread for each subset of images
        for i in range(self.threadCount):
            currentThread = threading.Thread(target=self.execute, args=(chunks[i],seedThreadCount,))
            currentThread.start()
            self.threads.append(currentThread)

    # Joins threaded processes to end execution
    # and prints stats
    def end(self):
        # Join threads once complete
        for thread in self.threads:
            thread.join()

        # Record the time it took to analyze the photos
        elapsed = time() - self.startTime
        avgTime = self.photoCount / elapsed
        hours = int(elapsed // 3600)
        secs = elapsed % 60
        mins = int((elapsed - secs - hours * 3600) // 60)

        # Alert user that the process is complete
        # and print program stats
        s = "s"
        if self.errors == 1:
            s = ""
        hoursMins = ""
        if (hours + mins):
            hoursMins = "{2:02d}:{3:02d} and "

        print(("\n{0} images analyzed in " + hoursMins + "{1:.2f} seconds").format(self.photosSeen, secs, hours, mins))
        print("Avg {0:.2f} images per second, {1} error{2}".format(avgTime, self.errors, s))


"""

*** BEGIN MAIN CODE EXECUTION ***

"""

if __name__ == "__main__":

    version = "1.5.5"
    dateModified = "July 11, 2024"

    measureSeeds = PhotoManager(version, dateModified)
    measureSeeds.begin() # Begin analysis using 6 threads
    measureSeeds.end()