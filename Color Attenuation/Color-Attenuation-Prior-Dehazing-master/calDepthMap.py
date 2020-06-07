import cv2
import scipy
import numpy as np

def calDepthMap(I, r):

    hsvI = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    s = hsvI[:,:,1] / 255.0
    v = hsvI[:,:,2] / 255.0
    #cv2.imshow("hsvI",hsvI)
    #cv2.waitKey()

    sigma = 0.041337
    sigmaMat = np.random.normal(0, sigma, (I.shape[0], I.shape[1]))

    output =  0.121779 + 0.959710 * v - 0.780245 * s + sigmaMat
    outputPixel = output
    output = scipy.ndimage.filters.minimum_filter(output,(r,r))
    outputRegion = output
    cv2.imwrite("data/vsFeature.jpg", outputRegion*255 )
    #cv2.imshow("outputRegion",outputRegion)
    #cv2.waitKey()
    return outputRegion, outputPixel
