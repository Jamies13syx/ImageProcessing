from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import imutils
import cv2


LicensePlate = namedtuple("LicensePlateRegion", ["success", "plate", "threshold", "candidates"])


class LicensePlateDetector:
    # store the image to detect license plates in, the minimum width and height of the
    # license plate region, the number of characters to be detected in the license plate,
    # and the minimum width of the extracted characters
    def __init__(self, image, minPlateWidth=60, minPlateHeight=20, numChars=7, minCharWidth=40):
        self.image = image
        self.minPlateWidth = minPlateWidth
        self.minPlateHeight = minPlateHeight
        self.numChars = numChars
        self.minCharWidth = minCharWidth

    def detectWrapper(self):
        # detect license plate regions in the image
        plateRegions = self.detectPlates()

        # loop over the license plate regions
        for plateRegion in plateRegions:
            # detect character candidates in the current license plate region
            licensePlate = self.detectCharacterCandidates(plateRegion)
            # only continue if characters were successfully detected
            if licensePlate.success:
                # scissorPlate the candidates into characters
                chars = self.plateScissor(licensePlate)
                # yield a tuple of the license plate region and the characters
                yield (plateRegion, chars)

    def detectPlates(self):
        # initialize the rectangular and square kernels to be applied to the image,
        # then initialize the list of license plate regions
        rectangleKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        regions = []

        # convert the image to grayscale and apply the blackhat operation
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blackhat_image = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, rectangleKernel)

        # compute the Scharr gradient representation of the blackhated image and scale the
        # resulting image into the range [0, 255]
        gradX = cv2.Sobel(blackhat_image,
                          ddepth=cv2.CV_32F,
                          dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

        # blur the gradient representation, apply a closing operating, and threshold the image using Otsu's method
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectangleKernel)
        threshold = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # perform a series of erosion and dilation on the image
        threshold = cv2.erode(threshold, None, iterations=2)
        threshold = cv2.dilate(threshold, None, iterations=2)

        # find regions in the image that are light
        light = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, squareKernel)
        light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]

        # take the bitwise 'and' between the 'light' regions of the image, then perform another series of erosions and dilations
        threshold = cv2.bitwise_and(threshold, threshold, mask=light)
        threshold = cv2.dilate(threshold, None, iterations=2)
        threshold = cv2.erode(threshold, None, iterations=1)

        # find contours in the threshold applied image
        contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in contours:
            # grab the bounding box associated with the contour and compute the area and aspect ratio
            (width, height) = cv2.boundingRect(c)[2:]
            aspectRatio = width / float(height)

            # calculate extent for additional filtering
            shapeArea = cv2.contourArea(c)
            boundingboxArea = width * height
            extent = shapeArea / float(boundingboxArea)
            extent = int(extent * 100) / 100

            # compute the rotated bounding box of the region
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)

            # ensure the aspect ratio, width, and height of the bounding box fall within tolerable limits, then update the list of license plate regions
            if height > self.minPlateHeight and width > self.minPlateWidth and extent > 0.4 and (2.5 < aspectRatio < 5.5):
                regions.append(box)

        # return the list of license plate regions
        return regions

    def detectCharacterCandidates(self, region):
        # apply a 4-point transform to extract the license plate
        plate = perspective.four_point_transform(self.image, region)
        cv2.imshow("Perspective Transform Result", imutils.resize(plate, width=400))

        # extract the Value component from the HSV color space and apply adaptive threshold to reveal the characters on the license plate
        V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 17, offset=15, method="gaussian")
        threshold = (V > T).astype("uint8") * 255
        threshold = cv2.bitwise_not(threshold)

        # resize the license plate region to a proper size
        plate = imutils.resize(plate, width=400)
        threshold = imutils.resize(threshold, width=400)
        cv2.imshow("Thresholding Result", threshold)

        # perform a connected components analysis and initialize the mask to store the locations of the character candidates
        labels = measure.label(threshold, neighbors=8, background=0)
        charCandidates = np.zeros(threshold.shape, dtype="uint8")

        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue

            # otherwise, construct the label mask to display only connected components for the current label, then find contours in the label mask
            labelMask = np.zeros(threshold.shape, dtype="uint8")
            labelMask[labels == label] = 255
            contours, _ = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # ensure at least one contour was found in the mask
            if len(contours) > 0:
                # grab the largest contour which corresponds to the component in the mask, then grab the bounding box for the contour
                c = max(contours, key=cv2.contourArea)
                (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

                # compute the aspect ratio, solidity, and height ratio for the component
                aspectRatio = boxW / float(boxH)
                solidity = cv2.contourArea(c) / float(boxW * boxH)
                heightRatio = boxH / float(plate.shape[0])

                # determine if the aspect ratio, solidity, and height of the contour pass the rules tests
                keepAspectRatio = aspectRatio < 1.0
                keepSolidity = solidity > 0.10
                keepHeight = 0.4 < heightRatio < 0.8

                # check to see if the component passes all the tests
                if keepAspectRatio and keepSolidity and keepHeight:
                    # compute the convex hull of the contour and draw it on the character candidates mask
                    hull = cv2.convexHull(c)
                    cv2.drawContours(charCandidates, [hull], -1, 255, -1)

        # clear pixels that touch the borders of the character candidates mask and detect contours in the candidates mask
        charCandidates = segmentation.clear_border(charCandidates)
        contours, _ = cv2.findContours(charCandidates.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("Original Candidates Result", charCandidates)

        # if there are more character candidates than the supplied number, then prune the candidates
        # if len(contours) > self.numChars:
        #     (charCandidates, contours) = self.pruneCandidates(charCandidates, contours)
        #     cv2.imshow("Pruned Candidates Result", charCandidates)

        # take bitwise AND of raw thresholded image and character candidates to get a more clean segmentation of the characters
        threshold = cv2.bitwise_and(threshold, threshold, mask=charCandidates)
        cv2.imshow("Char Threshold Result", threshold)

        # return the license plate region object containing the license plate, the thresholded license plate, and the character candidates
        return LicensePlate(success=len(contours) == self.numChars, plate=plate, threshold=threshold,
                            candidates=charCandidates)

    def pruneCandidates(self, charCandidates, contours):
        # initialize the pruned candidates mask and the list of dimensions
        prunedCandidates = np.zeros(charCandidates.shape, dtype="uint8")
        dimensions = []

        # loop over the contours
        for c in contours:
            # compute the bounding box for the contour and update the list of dimensions
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            dimensions.append(boxY + boxH)

        # convert the dimensions into a NumPy array and initialize the list of differences
        # and selected contours
        dimensions = np.array(dimensions)
        differences = []
        selected = []

        # loop over the dimensions
        for i in range(0, len(dimensions)):
            # compute the sum of differences between the current dimension and and all other dimensions, then update the differences list
            differences.append(np.absolute(dimensions - dimensions[i]).sum())

        # find the top number of candidates with the most similar dimensions and loop over
        # the selected contours
        for i in np.argsort(differences)[:self.numChars]:
            # draw the contour on the pruned candidates mask and add it to the list of selected contours
            cv2.drawContours(prunedCandidates, [contours[i]], -1, 255, -1)
            selected.append(contours[i])

        # return a tuple of the pruned candidates mask and selected contours
        return prunedCandidates, selected

    def plateScissor(self, plate):
        # detect contours in the candidates and initialize the list of bounding boxes and list of extracted characters
        contours, _ = cv2.findContours(plate.candidates.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        chars = []

        # loop over the contours
        for c in contours:
            # compute the bounding box for the contour while maintaining the minimum width
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            dX = min(self.minCharWidth, self.minCharWidth - boxW) // 2
            boxX -= dX
            boxW += (dX * 2)

            # update the list of bounding boxes
            boxes.append((boxX, boxY, boxX + boxW, boxY + boxH))

        # sort the bounding boxes from left to right
        boxes = sorted(boxes, key=lambda b: b[0])

        # loop over the started bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # extract the ROI from the thresholded license plate and update the characters list
            chars.append(plate.threshold[startY: endY, startX: endX])

        # return the list of characters
        return chars
