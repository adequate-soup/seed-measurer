#!/bin/python3

# Shebang that points to your python interpreter
# This should be changed based on your python environent
# and must remain on the program's first line

"""
 Benjamin Sims
 benjamis20@berkeley.edu

 Blackman Lab @ UC Berkeley
"""

# Import necessary packages
from time import time
from math import log10, hypot
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import argparse
import cv2
import os

author = "Benjamin Sims"
progName = "SeedMeasure"
ver = "1.8.0"
dateModified = "January 30, 2026"

progDescription = "Seed Measure takes an image of seeds on a bright background and measures each seed's\n \
    area, length, and width to a CSV file. Your seeds must be pictured inside a box of known\n \
    width. The program will attempt to correct for perspecitve skew based on the shape of your box."

fileFormats = ["jpeg", "jpg", "tiff", "gif", "png", "heic", "heif"]


# *** HELPER FUNCTIONS ***:

# Takes four unordered corner points of a quadrilateral,
# puts them into a consistent order to know which corner is which
def order_points(pts):
    # pts: (4, 2) array
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

# Checks if a contour is within a boundry
def contour_inside_bounds(contour, margin, bounds):
    x, y, w, h = cv2.boundingRect(contour)
    return (
        x > margin and y > margin and
        x + w < bounds["x"] and y + h < bounds["y"]
    )

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5,
            (ptA[1] + ptB[1]) * 0.5)

# Finds the width of a square contour
def box_pixel_width_from_contour(boxContour):
    box = cv2.minAreaRect(boxContour)
    box = cv2.boxPoints(box)
    box = order_points(box.astype("float32"))

    (tl, tr, br, bl) = box
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    return hypot(trbrX - tlblX, trbrY - tlblY)

# Finds the length and width of a contour (used for seed contours)
def seed_dimensions_from_contour(contour):

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = order_points(box.astype("float32"))

    (tl, tr, br, bl) = box

    # Midpoints of opposite edges
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # Distances between midpoint pairs
    dA = hypot(blbrX - tltrX, blbrY - tltrY)
    dB = hypot(trbrX - tlblX, trbrY - tlblY)

    return min(dA, dB), max(dA, dB)

# *** PROCEDURAL STEPS ***

# 1. Parse command line arguments
def parse_args():

    global progName, ver, author, dateModified, progDescription

    ap = argparse.ArgumentParser(prog=progName,
    description=progDescription)

    ap.add_argument("-i", "--indir", required=True, \
                    help="Path to an image or a directory of images.")
    ap.add_argument("-o", "--outdir", required=True, \
                    help="Output directory for processed images and CSV files.")
    ap.add_argument("-w", "--width", type=float, required=True, \
                    help="Width of the box around seeds in millimeters.")
    ap.add_argument("-c", "--cutoff", type=float, default=0, \
                    help="Seeds smaller than this size in square mm will be ignored (default: 0).")
    ap.add_argument("-e", "--erode", type=int, default=0, \
                    help="Perform morphological opening (erode then dilate) e number of iterations to separate touching seeds (default: 0).")
    ap.add_argument("-m", "--margin", type=float, default=5, \
                    help="Percent margin inside the box to ignore (default: 5).")
    ap.add_argument("-b", "--blocksize", type=int, default=599, \
                    help="Adaptive thresholding block size (odd number; default: 599).")
    ap.add_argument("-k", "--constant", type=float, default=20, \
                    help="Adaptive thresholding constant (default: 20).")
    ap.add_argument("-f", "--filtering", type=float, default=3, \
                    help="Standard deviation for outlier filtering (set to 0 to disable; default: 3).")
    ap.add_argument("-t", "--threads", type=int, default=8, \
                    help="Number of threads to process images (default: 8).")
    ap.add_argument("-v", "--version", action="store_true", \
                    help="Show version info and exit.")

    args = vars(ap.parse_args())

    if args["version"]:
        print(f"{progName} {ver} by {author} ({dateModified})")
        exit(0)

    if not args["width"] or args["width"] <= 0:
        ap.error("argument -w/--width is required and must be a positive number.")

    # Convert any negative values to positive
    for key in ["threads", "width", "margin", "constant", "filtering", "cutoff", "erode", "blocksize"]:
        args[key] = abs(args[key])

    # Make sure blocksize is odd
    args["blocksize"] = args["blocksize"] if args["blocksize"] % 2 == 1 else args["blocksize"] + 1

    return args["threads"], args["width"], args["outdir"], args["indir"], args["margin"], \
        args["blocksize"], args["constant"], args["filtering"], args["cutoff"], args["erode"]

# 2. Load an image from the specified file path
# Read the image into opencv and return that object
def load_image(inDir, imgName):
       
        imgOriginal = cv2.imread(os.path.join(inDir, imgName))
        if imgOriginal is None:
            raise FileNotFoundError(f"Could not read image file: {imgName}")
        return imgOriginal

# 3. Find the quadrilateral bounding box in the image
def find_box(imgOriginal):
        
        gray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise Exception("No contours found to identify a bounding box.")
        
        # Find the bounding box (second largest quadrilateral)
        largest = max(cnts, key=cv2.contourArea)
        cnts_wo_largest = (c for c in cnts if c is not largest)
        boxContour = max(cnts_wo_largest, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(boxContour, True)
        approx = cv2.approxPolyDP(boxContour, epsilon, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)
        else:
            raise Exception("Unable to find a four-sided bounding box.")

# 4. Crop original input image and correct for skew to make a perfect square,
# using printed bounding box found in previous step
def crop_image(imgOriginal, boxPts):
    
    rect = order_points(boxPts)
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
    return cv2.warpPerspective(imgOriginal, M, (sideLength, sideLength))

# 5. Find all potential seeds in cropped image
def find_seed_contours(imgCropped, blockSize, constant, erode, marginPercent):
        gray = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY)
        warpedThresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, blockSize, constant)
        """
         If the erode flag is set, perform erosion followed by dilation.
         This "opening" operation separates touching contours and removes noise
         while better preserving the original size of the seeds.
        """
        image_for_contours = warpedThresh
        if erode > 0:
            kernel = np.ones((3, 3), np.uint8)
            eroded_image = cv2.erode(warpedThresh, kernel, iterations=erode)
            image_for_contours = cv2.dilate(eroded_image, kernel, iterations=erode)

        cnts, _ = cv2.findContours(image_for_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not len(cnts):
            raise Exception("No contours found in the cropped image.")
        
        # Find largest contour (bounding box), all other contours are seeds
        boxContour = max(cnts, key=cv2.contourArea)
        seedContours = [c for c in cnts if c is not boxContour]
        #cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        #boxContour = cnts[0]
        #seedContours = cnts[1:]

        margin = int((marginPercent / 100.0) * warpedThresh.shape[1])

        # Returns box contour, seed contours, margin to ignore, x and y bounds of space to analyze
        return boxContour, seedContours, margin, {"x":warpedThresh.shape[1] - margin, "y":warpedThresh.shape[0] - margin}

# 6. Measure all the seeds (area, length, and width for each valid seed contour)
def measure_seeds(seedContours, margin, bounds, pxPerMil):

    filteredSeedContours = []
    seedMeasurements = []
    
    # Don't measure seeds that fall outside of the margin
    for contour in seedContours:
        if not contour_inside_bounds(contour, margin, bounds):
            continue

        dA, dB = seed_dimensions_from_contour(contour)
        area = cv2.contourArea(contour) / pxPerMil ** 2
        metrics = [dA / pxPerMil, dB / pxPerMil]
        width, length = min(metrics), max(metrics)

        if not length or not width or not area:
            continue
        
        seedMeasurements.append([area, length, width])
        filteredSeedContours.append(contour)
    
    if not len(seedMeasurements):
        raise Exception("No seeds detected within supplied margin of image")

    return filteredSeedContours, seedMeasurements

# 7. Filter outlier seeds first by a minimum area cutoff, then by statistical outliers.
def filter_seeds(seedMeasurements, seedContours, cutoff, filteringStDevs):

    passed_area_check = seedMeasurements
    passed_area_contours = seedContours

    # Filter by minimum area cutoff first
    if cutoff > 0:
        passed_area_check = []
        passed_area_contours = []
        for seed, contour in zip(seedMeasurements, seedContours):
            area = seed[0] # Area is the first item in the list
            if area >= cutoff:
                passed_area_check.append(seed)
                passed_area_contours.append(contour)

    # Perform statistical filtering
    # If stdevs is 0, this block is skipped, and all seeds that passed the area cutoff are kept.
    seedMeasurements_filtered = np.asarray(passed_area_check)
    filteredContours_final = passed_area_contours

    if filteringStDevs > 0:
        # Uses fancy numpy array magic
        data = np.asarray(passed_area_check, dtype=float)

        median = np.median(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = np.inf

        lower = median - filteringStDevs * std
        upper = median + filteringStDevs * std

        mask = np.all((data >= lower) & (data <= upper), axis=1)
        #mask = np.sum((data >= lower) & (data <= upper), axis=1) >= (data.shape[1] - 1)

        seedMeasurements_filtered = data[mask]
        idx = np.nonzero(mask)[0]
        filteredContours_final = [passed_area_contours[i] for i in idx]

    if not len(seedMeasurements_filtered):
        raise Exception("No seeds passed filtering or cutoff minima")

    # Return list of filtered seeds and contours
    return seedMeasurements_filtered, filteredContours_final

# 8. Create quality check image on the cropped box area using the list of seed contours
def draw_contours(imgCropped, seedContours, outDir, lineName, downsizeRatio=1, color="red"):
        
        if not len(seedContours):
            return 0
        
        resizedImage = imgCropped
        
        color_map = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0)}
        colorTuple = color_map.get(color.lower(), (0, 0, 255))

        cv2.drawContours(imgCropped, seedContours, -1, colorTuple, thickness=cv2.FILLED)

        if 0 < downsizeRatio < 1:
             resizedImage = cv2.resize(imgCropped, (0, 0), fx=downsizeRatio, fy=downsizeRatio, interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(os.path.join(outDir, (lineName + "_seeds.jpg")),resizedImage)

# 9. Write all measurements to a CSV file
def write_csv(outDir, lineName, seedMeasurements, headers=["line_name", "seeds_imaged", "area_mm2", "length_mm", "width_mm"]):

    path = os.path.join(outDir, lineName + ".csv")
    seedCount = len(seedMeasurements)
    four_decimals = "{:.4f}".format

    with open(path, "w", encoding="utf-8") as f:
        # Write header
        f.write(",".join(headers) + "\n")

        for seed_metrics in seedMeasurements:
            row = [lineName, str(seedCount)]
            row.extend(four_decimals(x) for x in seed_metrics)
            f.write(",".join(row) + "\n")

# 10. Worker to process a single image
def process_image(imgFile, inDir, outDir, width, marginPercent,
                  blockSize, constant, filteringStDevs, cutOff, erode):

    cv2.setNumThreads(1)

    # Run all procedural steps sequentially
    # See above comments for details
    try:
        lineName = os.path.splitext(os.path.basename(imgFile))[0]

        imgOriginal = load_image(inDir, imgFile)
        boundingBox = find_box(imgOriginal)
        imgCropped = crop_image(imgOriginal, boundingBox)

        boxContour, seedContours, marginPixels, bounds = find_seed_contours(
            imgCropped, blockSize, constant, erode, marginPercent
        )

        # Calculate millimeter-to-pixel ratio
        boxPixelWidth = box_pixel_width_from_contour(boxContour)
        pxPerMil = boxPixelWidth / width

        seedContours, seedMeasurements = measure_seeds(
            seedContours, marginPixels, bounds, pxPerMil
        )

        # Bug in filtering logic
        seedMeasurementsFiltered, seedContoursFiltered = filter_seeds(
            seedMeasurements, seedContours, cutOff, filteringStDevs
        )

        draw_contours(imgCropped, seedContoursFiltered, outDir, lineName)
        write_csv(outDir, lineName, seedMeasurementsFiltered)

        return True, imgFile, None

    except Exception as e:
        return False, imgFile, str(e)

# *** MAIN CODE EXECUTION ***
def main():

    # Parse arguments
    threadCount, width, outDir, inDir, marginPercent, \
        blockSize, constant, filteringStDevs, cutOff, erode \
        = parse_args()

    # Make output directory
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    # Check if input is a directory of a single file
    dirList = []
    if os.path.isdir(inDir):
        dirList = [f for f in os.listdir(inDir) if f.split(".")[-1].lower() in fileFormats and not f.startswith(".")]
    else:
        # Only read from files in an accepted file format
        if os.path.exists(inDir) and inDir.split(".")[-1].lower() in fileFormats:
            dirList.append(os.path.basename(inDir))
            inDir = os.path.dirname(inDir)

    # Raise an error if no compatible files are found
    if not dirList:
        raise Exception(f"No valid images found in the provided path: {inDir}")

    photoCount = len(dirList)
    digits = str(int(log10(photoCount)) + 1) if photoCount > 0 else "1"

    # Dont run more threads than there are images to analyze
    if photoCount < threadCount:
        threadCount = photoCount

    if threadCount == 0:
        raise Exception("No images to process.")
    
    print(f"Processing {photoCount} images using {threadCount} processes...\n")

    errors = 0
    completed = 0
    startTime = time()

    # Multiprocessing
    with ProcessPoolExecutor(max_workers=threadCount) as executor:
        futures = {
            executor.submit(
                process_image,
                imgFile, inDir, outDir, width, marginPercent,
                blockSize, constant, filteringStDevs, cutOff, erode
            ): imgFile
            for imgFile in dirList
        }

        for future in as_completed(futures):
            success, imgFile, err = future.result()
            completed += 1

            print(f"({completed:0{digits}d}/{photoCount}) {imgFile} analyzed sucessfully")

            if not success:
                errors += 1
                print(f"ERROR in {imgFile}: {err}")

    # Print stats after all processes are complete
    elapsed = time() - startTime
    avgTime = photoCount / elapsed if elapsed > 0 else 0
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = elapsed % 60
    s = "" if errors == 1 else "s"

    print("\n--- Analysis Complete ---")
    print(f"{photoCount} images analyzed in {hours:02d}h {mins:02d}m {secs:.2f}s")
    print(f"Average of {avgTime:.2f} images per second, with {errors} error{s}.")

# Call main()
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nA critical error occurred: {e}")