**Arguments**

Example usage: seed\_measure \-i /path/to/photos \-o /path/to/output \-w 100

| Argument: | Required / optional | Data type | Description |
| :---- | :---- | :---- | :---- |
| \-i / \--indir | Required | Path | The path to the directory of images to be processed or a path to the location of a single image |
| \-o / \--outdir | Required | Path | The path to the directory where SeedMeasure will output CSV files and quality check images |
| \-w / \--width | Required | Floating point | Width of the bounding box in which your seeds are imaged. SeedMeasure expects millimeters, but any unit of measurement can be supplied |
| \-m / \--margin | Optional (default: 5\) | Floating point | The inner margin as percentage of the bounding box width to ignore while looking for seeds |
| \-f / \--filtering | Optional (default: 3\) | Floating point | Number of standard deviations from the median to include in the final seed measurements |
| \-t / \--threads | Optional (default: 1\) | Integer | Number of threads to use for image processing. Each image is run on a single thread |
| \-c / \--cutoff | Optional (default: 0\) | Floating point | All seeds smaller than this provided area will be ignored |
| \-b / \--blocksize | Optional (default: 599\) | Integer (odd) | Size in pixels of the block used to detect average brightness of your image. Decrease this value when using lower resolution images. This should be an odd number |
| \-k / \--constant | Optional (default: 20\) | Floating point | Number to subtract from the brightness of each pixel while thresholding. This is the effective inverse of sensitivity where lower values will be more sensitive to objects with low contrast to the background |
| \-e / \--erode | Optional (default:0) | Integer | Number of erosion iterations to perform while thresholding. Increasing this value may help to differentiate between seeds that are clumped closely together. |
| \-h / \--help | Optional | Flag | Prints a help message and exits |
| \-v / \--version | Optional | Flag | Prints the script’s version and exits |

NOTE: the default settings of the optional arguments should work for most seeds 

**Seed Imaging**

Seeds must be imaged against a white sheet of photo paper within a printed reference box \[see fig\]. This box can be any size depending on your needs, but it must be a perfect square. Its outside dimensions are supplied (which SeedMeasure assumes to be in millimeters) as an argument using the \-w flag. The paper used for seed imaging should be placed on a lightbox in a darkened room to minimize shadows and ensure maximum contrast between seeds and background. Seeds should be spread apart manually before imaging to reduce clumping. SeedMeasure will exclude seeds that appear to be clumped, but minimizing the number of clumped seeds manually will improve accuracy. Images can be captured with a smartphone camera (ideally having a 12 MP sensor for medium to large-sized seeds, or a 50 MP sensor for *Arabidopsis thaliana* sized seeds or smaller).

The filename of each input image will be used as the line name for measurements in the output CSV file (e.g., aribidopsis\_101.jpg is recorded as arabidopsis\_101 in line\_name column for each row of the output). You can supply these images individually to the program using the \-i flag, or you can put all your images in the same directory to analyze them in a single batch. 

**Image Processing and Measurement**

SeedMeasure detects the seed bounding box by searching for the largest square (by area) in your image. Make sure any labels and seed packets you image are smaller than the bounding box.

Seeds located too close to the border of the box (within 5% of the edge by default) are excluded to minimize measurement error due to out-of-focus boundaries, but the user can change or remove this margin by supplying a percentage of the total box width as the \-m flag. After every seed is detected, SeedMeasure records cross-sectional area, length (the maximum distance between two points on the seed’s perimeter), and width (the minimum distance between two points on the seed’s perimeter). Each measurement is written to a comma-separated values (CSV) file for downstream analysis with the total seed count per image recorded in its own column.

To verify accuracy, SeedMeasure also produces a quality check (QC) image with measured seeds highlighted in red while all other objects remain gray. 

**Automated Filtering**

SeedMeasure automatically filters out objects whose area, length, width, or length-to-width ratio fall more than three standard deviations from the median of the sample. These objects are discarded from the dataset. Automated filtering may not be appropriate if your seeds vary greatly in size or shape. Additionally, excess debris in images will result in less accurate filtering. Users may optionally disable or change the number of standard deviations used for this filtering feature with the \-f flag.

**Performance and Reproducibility**

When supplying a directory of many images at once, users can supply the number of threads to use with the \-t flag, allowing SeedMeasure to process multiple images concurrently. The speed of image analysis scales proportionally to the number of threads available.

**Troubleshooting**

Images taken at varying resolutions and brightnesses may require fine tuning of Seed Measurer’s thresholding settings. Using default settings, users may find that SeedMeasure’s thresholding is too sensitive, which will often result in details from the imaging background’s texture registering as seeds in the QC image. To fix this, try increasing the value given to the \-k flag in increments of 1 until the background is no longer highlighted in red in your QC images. If these issues persist, try providing a minimum area cutoff using the \-c flag set to a value just below the expected minimum cross-sectional area of your seed.

If SeedMeasure detects no seeds in your image, try decreasing the constant until your seeds show up in the QC image. Filtering settings may also help if extra large, small, or oddly shaped seeds are flagged as outliers and excluded from analysis. Supplying a lower filtering level using the \-f flag or setting filtering to 0 to disable it completely may resolve these issues. If clumped seeds or debris are not excluded from analysis, try increasing the filtering level or manually separate your seeds, remove excess debris, and re-take your image.

**Data Availability**

SeedMeasure is open-sourced (MIT License) and is available on GitHub ([adequate-soup/seed-measurer: Python script for counting and measuring the area, width, and length of small seeds](https://github.com/adequate-soup/seed-measurer)). Please open an issue on GitHub if you run into any problems or bugs in the script.