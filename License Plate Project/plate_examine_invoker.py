from imutils import paths
import traceback
import argparse
import imutils
import numpy as np
import random
import cv2
import os

# construct the argument parser and parse the argument
from license_plate_util import LicensePlateDetector

argumentParser = argparse.ArgumentParser()
argumentParser.add_argument("-i", "--images", required=True, help="Please input path to the images to be examined")
argumentParser.add_argument("-o", "--outputs", required=True, help="Please input path to the output directory")
args = vars(argumentParser.parse_args())

# randomly select a portion of the images and initialize the dictionary of character counts
imagePaths = list(paths.list_images(args["images"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[:int(len(imagePaths) * 0.8)]
counts = {}

# loop over the images
for imagePath in imagePaths:
    # show the image path
    print("Program is trying to detect {}".format(imagePath))

    try:
        # load the image
        image = cv2.imread(imagePath)

        # if the width is greater than 640 pixels, then resize the image
        if image.shape[1] > 640:
            image = imutils.resize(image, width=640)

        # initialize the license plate detector and detect character on the license plate
        licencePlateDetector = LicensePlateDetector(image, numChars=7)
        plates = licencePlateDetector.detectWrapper()

        # loop over the license plates
        for (box, chars) in plates:

            box = np.array(box).reshape((-1, 1, 2)).astype(np.int32)

            # draw the bounding box surrounding the license plate and display it for reference purposes
            plate = image.copy()
            cv2.drawContours(plate, [box], -1, (255, 0, 0), 2)
            cv2.imshow("License Plate Result", plate)

            index = 0
            # loop over thg characters
            for char in chars:
                index = index + 1
                # display the character and wait for a keypress
                # cv2.imshow("Char Result", char)
                # grab the key that was pressed and construct the path to the output directory
                key = imagePath.replace('\\', '_').replace('.', '_')
                dirPath = "{}/{}".format(args["outputs"], key)

                # if the output directory does not exist, create it
                if not os.path.exists(dirPath):
                    os.makedirs(dirPath)

                path = "{}/{}.png".format(dirPath, str(index).zfill(5))
                cv2.imwrite(path, char)
            cv2.waitKey(0)

    # an unknown error happens for this particular image, so do not process it
    # and display a traceback for debugging purposes
    except KeyboardInterrupt:
        break
    except:
        print(traceback.format_exc())
        print("Program is detecting error happens at {}".format(imagePath))
