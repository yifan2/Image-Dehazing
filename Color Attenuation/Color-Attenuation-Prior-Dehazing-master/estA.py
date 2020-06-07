import numpy as np
import cv2

import scipy.io as sio
import numpy as np

def estA(img, Jdark):
    
    #Estimate Airlignt of image I
    h,w,c = img.shape
    
    if img.dtype == np.uint8:
        img = np.float32(img) / 255
    
    # Compute number for 0.1% brightest pixels
    n_bright = int(np.ceil(0.001*h*w))
    #  Loc contains the location of the sorted pixels
    reshaped_Jdark = Jdark.reshape(1,-1)
    Y = np.sort(reshaped_Jdark) 
    Loc = np.argsort(reshaped_Jdark)
    
    # column-stacked version of I
    Ics = img.reshape(1, h*w, 3)
    ix = img.copy()
    dx = Jdark.reshape(1,-1)
    
    # init a matrix to store candidate airlight pixels
    Acand = np.zeros((1, n_bright, 3), dtype=np.float32)
    # init matrix to store largest norm arilight
    Amag = np.zeros((1, n_bright, 1), dtype=np.float32)
    
    # Compute magnitudes of RGB vectors of A
    for i in xrange(n_bright):
        x = Loc[0,h*w-1-i]
        ix[x/w, x%w, 0] = 0
        ix[x/w, x%w, 1] = 0
        ix[x/w, x%w, 2] = 1
        
        Acand[0, i, :] = Ics[0, Loc[0, h*w-1-i], :]
        Amag[0, i] = np.linalg.norm(Acand[0,i,:])
    
    # Sort A magnitudes
    reshaped_Amag = Amag.reshape(1,-1)
    Y2 = np.sort(reshaped_Amag) 
    Loc2 = np.argsort(reshaped_Amag)
    # A now stores the best estimate of the airlight
    if len(Y2) > 20:
        A = Acand[0, Loc2[0, n_bright-19:n_bright],:]
    else:
        A = Acand[0, Loc2[0,n_bright-len(Y2):n_bright],:]
    
    # finds the max of the 20 brightest pixels in original image
    print A

    #cv2.imshow("brightest",ix)
    #cv2.waitKey()
    cv2.imwrite("data/position_of_the_atmospheric_light.png", ix*255)
    
    return A

if __name__ == "__main__":
    data = sio.loadmat("para.mat")
    I = data["I"]
    dR = data["dR"]
    a = estA(I,dR)
    print a
