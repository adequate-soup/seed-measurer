#!/bin/python3

# Shebang that points to your python interpreter
# This should be changed based on your python environent
# and must remain on the program's first line

"""
 Benjamin Sims
 benjamis20@berkeley.edu

 Blackman Lab @ UC Berkeley
 Last Updated January 28, 2026

 Seed Measure takes an image of seeds on a bright background and measures each seed's
 area, length, and width to a CSV file. Your seeds must be pictured inside a box of known
 width. The program will attempt to correct for perspecitve skew based on the shape of your box.
"""

# Import necessary packages
from imutils import perspective
from time import time
from math import log10, hypot
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
    def __init__(self, inDir, outDir, imgName, boxWidth, marginPercent, blockSize, constant, filtering, cutoff, erode):
        # --- Constructor now only initializes state, does not run processing ---
        self.inDir = inDir
        self.outDir = outDir
        self.imgName = imgName
        self.boxWidth = boxWidth
        self.marginPercent = marginPercent / 100
        self.blockSize = blockSize
        self.constant = constant
        self.filtering = filtering
        self.cutoff = cutoff
        self.erode = erode

        # Initialize placeholder variables
        self.seedCount = 0
        self.imgOriginal = None
        self.imgCropped = None
        self.boxPts = None
        self.boxContour = None
        self.seedContours = []
        self.seedMeasurements = []
        self.pxPerMil = 0
        self.xBound = 0
        self.yBound = 0
        self.margin = 0

        # --- MODIFIED: Simplified filename processing ---
        # Set the line_name from the image filename and define CSV headers
        self.headers = ["line_name", "seeds_imaged", "area_mm2", "length_mm", "width_mm"]
        # Gets the filename without the extension
        self.line_name = os.path.splitext(os.path.basename(imgName))[0]


    def process(self):
        """
        Public method to run the entire image analysis pipeline.
        """
        self._load_image()
        self._find_box()
        self._crop_image()
        self._find_seed_contours()
        self._set_pixel_metric()
        self._measure_seeds()
        self.filter_seeds()
        return self.seedCount

    def _load_image(self):
        """Loads the image from the specified file path."""
        self.imgOriginal = cv2.imread(os.path.join(self.inDir, self.imgName))
        if self.imgOriginal is None:
            raise FileNotFoundError(f"Could not read image file: {self.imgName}")

    def _find_box(self):
        """Finds the quadrilateral bounding box in the image."""
        gray = cv2.cvtColor(self.imgOriginal, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[1:]

        if not cnts:
            raise Exception("No contours found to identify a bounding box.")
        
        boxContour = cnts[0]
        epsilon = 0.02 * cv2.arcLength(boxContour, True)
        approx = cv2.approxPolyDP(boxContour, epsilon, True)

        if len(approx) == 4:
            self.boxPts = approx.reshape(4, 2)
        else:
            raise Exception("Unable to find a four-sided bounding box.")

    def _crop_image(self):
        """
        Crops the image to the bounding box, corrects for perspective, and makes it square.
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = self.boxPts.sum(axis=1)
        rect[0] = self.boxPts[np.argmin(s)]
        rect[2] = self.boxPts[np.argmax(s)]

        diff = np.diff(self.boxPts, axis=1)
        rect[1] = self.boxPts[np.argmin(diff)]
        rect[3] = self.boxPts[np.argmax(diff)]

        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        sideLength = max(maxWidth, maxHeight)

        dst = np.array([
            [0, 0],
            [sideLength - 1, 0],
            [sideLength - 1, sideLength - 1],
            [0, sideLength - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        self.imgCropped = cv2.warpPerspective(self.imgOriginal, M, (sideLength, sideLength))

    def _find_seed_contours(self):
        """Finds all potential seed contours in the cropped image."""
        gray = cv2.cvtColor(self.imgCropped, cv2.COLOR_BGR2GRAY)
        warpedThresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, self.blockSize, self.constant)
        
        # --- UPDATED: Optional morphological opening step ---
        # If the erode flag is set, perform erosion followed by dilation.
        # This "opening" operation separates touching contours and removes noise
        # while better preserving the original size of the seeds.
        image_for_contours = warpedThresh
        if self.erode > 0:
            kernel = np.ones((3, 3), np.uint8)
            eroded_image = cv2.erode(warpedThresh, kernel, iterations=self.erode)
            image_for_contours = cv2.dilate(eroded_image, kernel, iterations=self.erode)

        cnts = cv2.findContours(image_for_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        if not cnts:
            raise Exception("No contours found in the cropped image.")

        self.boxContour = cnts[0]
        self.seedContours = cnts[1:]
        self.margin = int(self.marginPercent * warpedThresh.shape[1])
        self.xBound = warpedThresh.shape[1] - self.margin
        self.yBound = warpedThresh.shape[0] - self.margin

    def _set_pixel_metric(self):
        """Calculates the pixels-per-millimeter ratio from the bounding box."""
        box = cv2.minAreaRect(self.boxContour)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        (tl, tr, br, bl) = box
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        dB = hypot(trbrX - tlblX, trbrY - tlblY)
        self.pxPerMil = dB / self.boxWidth

    def _measure_seeds(self):
        """
        Calculates area, length, and width for each valid seed contour.
        """
        filteredContours = []
        
        for contour in self.seedContours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (x > self.margin and y > self.margin and (x + w) < self.xBound and (y + h) < self.yBound):
                continue

            box = cv2.minAreaRect(contour)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            dA = hypot(blbrX - tltrX, blbrY - tltrY)
            dB = hypot(trbrX - tlblX, trbrY - tlblY)


            area = cv2.contourArea(contour) / self.pxPerMil ** 2
            metrics = [dA / self.pxPerMil, dB / self.pxPerMil]
            width, length = min(metrics), max(metrics)

            if not length or not width or not area:
                continue
            
            self.seedMeasurements.append([area, length, width])
            filteredContours.append(contour)
        
        self.seedContours = filteredContours
        self.seedCount = len(self.seedMeasurements)
    
    def filter_seeds(self):
        """
        Filters seeds first by a minimum area cutoff, then by statistical outliers.
        """
        if not self.seedMeasurements:
            return 0

        # --- Step 1: Filter by minimum area cutoff first ---
        # This is more efficient as it reduces the list size before the next step.
        passed_area_check = []
        passed_area_contours = []
        for seed, contour in zip(self.seedMeasurements, self.seedContours):
            area = seed[0] # Area is the first item in the list
            if area >= self.cutoff:
                passed_area_check.append(seed)
                passed_area_contours.append(contour)

        if not passed_area_check:
            self.seedMeasurements = []
            self.seedContours = []
            self.seedCount = 0
            return 0

        # --- Step 2: Conditionally perform statistical filtering ---
        # If stdevs is 0, this block is skipped, and all seeds that passed
        # the area cutoff are kept.
        if self.filtering > 0:
            metrics = []
            for i in range(3):  # For area, length, and width
                # Build the array from the pre-filtered list of seeds
                arr = [dataArr[i] for dataArr in passed_area_check]
                std = np.std(arr)
                median = np.median(arr)
                metrics.append([median - self.filtering * std, median + self.filtering * std])

            seedMeasurements_filtered = []
            filteredContours_final = []
            for seed, contour in zip(passed_area_check, passed_area_contours):
                record = True
                for i in range(3):
                    # The area cutoff check is no longer needed here
                    if not (metrics[i][0] < seed[i] < metrics[i][1]):
                        record = False
                        break
                if record:
                    seedMeasurements_filtered.append(seed)
                    filteredContours_final.append(contour)
                
            self.seedMeasurements = seedMeasurements_filtered
            self.seedContours = filteredContours_final
        else:
            # If filtering is disabled, use the results from the cutoff step
            self.seedMeasurements = passed_area_check
            self.seedContours = passed_area_contours
            
        self.seedCount = len(self.seedMeasurements)
        return self.seedCount

    def draw_contours(self, downsizeRatio=1, color="red"):
        """Draws the final, filtered seed contours on the cropped image and saves it."""
        color_map = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0)}
        colorTuple = color_map.get(color.lower(), (0, 0, 255))

        cv2.drawContours(self.imgCropped, self.seedContours, -1, colorTuple, thickness=cv2.FILLED)

        if 0 < downsizeRatio < 1:
             self.imgCropped = cv2.resize(self.imgCropped, (0, 0), fx=downsizeRatio, fy=downsizeRatio, interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(os.path.join(self.outDir, (self.line_name + "_seeds.jpg")), self.imgCropped)

    def write_csv(self):
        """Writes all measured seed data to a CSV file."""
        # --- MODIFIED: Simplified CSV output ---
        with open(os.path.join(self.outDir, (self.line_name + ".csv")), "w", newline="") as csvFile:
            seedWriter = csv.writer(csvFile)
            seedWriter.writerow(self.headers)

            four_decimals = lambda num: format(num, ".4f")

            for seed_metrics in self.seedMeasurements:
                # The row now only contains line_name and seed count before the metrics
                row = [self.line_name, self.seedCount]
                row.extend(map(four_decimals, seed_metrics))
                seedWriter.writerow(row)

class PhotoManager:
    def __init__(self, progVersion, dateModified):
        self.progName = "Seed Measurer"
        self.ver = progVersion
        self.author = "Benjamin Sims"
        self.dateModified = dateModified
        self.startTime = time()
        self._parse_args()

        self.fileFormats = ["jpeg", "jpg", "tiff", "gif", "png", "heic"]
        self.dirList = []
        if os.path.isdir(self.inDir):
            self.dirList = [f for f in os.listdir(self.inDir) if f.split(".")[-1].lower() in self.fileFormats and not f.startswith(".")]
        else:
            if os.path.exists(self.inDir) and self.inDir.split(".")[-1].lower() in self.fileFormats:
                self.dirList.append(os.path.basename(self.inDir))
                self.inDir = os.path.dirname(self.inDir)

        if not self.dirList:
            raise Exception(f"No valid images found in the provided path: {self.inDir}")
        
        self.photoCount = len(self.dirList)
        self.photosSeen = 0
        self.errors = 0
        self.digits = str(int(log10(self.photoCount)) + 1) if self.photoCount > 0 else "1"
        self.threads = []
        self.lock = threading.Lock()

    def _parse_args(self):
        ap = argparse.ArgumentParser(prog=self.progName,
            description="Measures seeds in images within a box of known width.")
        ap.add_argument("-i", "--indir", required=True, help="Path to an image or a directory of images.")
        ap.add_argument("-o", "--outdir", required=True, help="Output directory for processed images and CSV files.")
        ap.add_argument("-w", "--width", type=float, required=True, help="Width of the box around seeds in millimeters.")
        ap.add_argument("-c", "--cutoff", type=float, default=0, help="Seeds smaller than this size in square mm will be ignored (default: 0).")
        ap.add_argument("-e", "--erode", type=int, default=0, help="Perform morphological opening (erode then dilate) e number of iterations to separate touching seeds (default: 0).")
        ap.add_argument("-m", "--margin", type=float, default=5, help="Percent margin inside the box to ignore (default: 5).")
        ap.add_argument("-b", "--blocksize", type=int, default=599, help="Adaptive thresholding block size (odd number; default: 599).")
        ap.add_argument("-k", "--constant", type=float, default=20, help="Adaptive thresholding constant (default: 20).")
        ap.add_argument("-f", "--filtering", type=float, default=3, help="Standard deviation for outlier filtering (set to 0 to disable; default: 3).")
        ap.add_argument("-t", "--threads", type=int, default=8, help="Number of threads to process images (default: 8).")
        ap.add_argument("-v", "--version", action="store_true", help="Show version info and exit.")
        
        args = vars(ap.parse_args())

        if args["version"]:
            self.version()

        if not args["width"] or args["width"] <= 0:
            ap.error("argument -w/--width is required and must be a positive number.")

        self.threadCount = abs(args["threads"])
        self.boxWidth = args["width"]
        self.outDir = args["outdir"]
        self.inDir = args["indir"]
        self.marginPercent = args["margin"]
        self.blockSize = abs(args["blocksize"]) if abs(args["blocksize"]) % 2 == 1 else abs(args["blocksize"]) + 1
        self.constant = abs(args["constant"])
        self.filtering = abs(args["filtering"])
        self.cutoff = abs(args["cutoff"])
        self.erode = abs(args["erode"])

        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)

    def version(self):
        print(f"{self.progName} {self.ver} by {self.author} ({self.dateModified})")
        exit(0)

    def execute(self, images):
        for file in images:
            self.lock.acquire()
            self.photosSeen += 1
            print(f"\n({self.photosSeen:0{self.digits}d}/{self.photoCount:0{self.digits}d}) Reading from {file}")
            self.lock.release()

            try:
                seedReader = PhotoReader(self.inDir, self.outDir, file, self.boxWidth, self.marginPercent,
                                         self.blockSize, self.constant, self.filtering, self.cutoff, self.erode)
                seed_count = seedReader.process()
                
                if seed_count > 0:
                    seedReader.draw_contours()
                    seedReader.write_csv()
                else:
                    self.lock.acquire()
                    print(f"WARNING in {file}: No seeds found after processing and filtering.")
                    self.lock.release()
            
            except Exception as e:
                self.lock.acquire()
                print(f"ERROR in {file}: {e}")
                self.errors += 1
                self.lock.release()

    def begin(self):
        if self.photoCount < self.threadCount:
            self.threadCount = self.photoCount

        if self.threadCount == 0:
            print("No images to process.")
            return

        chunks = [[] for _ in range(self.threadCount)]
        for i, file in enumerate(self.dirList):
            chunks[i % self.threadCount].append(file)

        for i in range(self.threadCount):
            currentThread = threading.Thread(target=self.execute, args=(chunks[i],))
            currentThread.start()
            self.threads.append(currentThread)

    def end(self):
        for thread in self.threads:
            thread.join()

        elapsed = time() - self.startTime
        avgTime = self.photoCount / elapsed if elapsed > 0 else 0
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        secs = elapsed % 60
        s = "" if self.errors == 1 else "s"

        print("\n--- Analysis Complete ---")
        print(f"{self.photoCount} images analyzed in {hours:02d}h {mins:02d}m {secs:.2f}s")
        print(f"Average of {avgTime:.2f} images per second, with {self.errors} error{s}.")

if __name__ == "__main__":
    version = "1.7.4" # Updated version
    dateModified = "January 28, 2026"

    try:
        measureSeeds = PhotoManager(version, dateModified)
        measureSeeds.begin()
        measureSeeds.end()
    except Exception as e:
        print(f"\nA critical error occurred: {e}")