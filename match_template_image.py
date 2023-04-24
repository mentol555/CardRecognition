import string
import warnings
import cv2
import numpy as np
import pytesseract
from numpy import array

from scipy import ndimage

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

warnings.filterwarnings('ignore')

# TM_SQDIFF_NORMED metódus
def match_template(image, template):
    # (W - w + 1 , H - h + 1)
    # W->image width  w->template width
    W = image.shape[0]
    H = image.shape[1]
    w = template.shape[0]
    h = template.shape[1]
    image = image.astype(np.int64)
    template = template.astype(np.int64)
    result = np.zeros((W - w + 1, H - h + 1), dtype=np.float64)
    for i in range(0, result.shape[0]):
        for j in range(0, result.shape[1]):
            szamlalo = 0
            nevezo_1 = 0
            nevezo_2 = 0
            for x in range(0, w):
                for y in range(0, h):
                    szamlalo += (template[x, y] - image[i+x, j+y])**2
                    nevezo_1 += template[x, y]**2
                    nevezo_2 += image[i+x, j+y]**2
            result[i, j] = szamlalo / (np.sqrt(nevezo_1 * nevezo_2))
            if(result[i, j]>1.0): result[i,j] = 1.0
    return result

# multi scale symbol matching
# 3 méretarányt kipróbál, minta szimbólumot illeszt,
# a legjobban illeszkedőt választja
def matchSymbol(image, template):
    minVal = 100
    for scale in (12, 15, 17):
        # rescale, keep the ratio!
        (h, w) = image.shape[:2]
        height = template.shape[0] * scale
        r = height / float(h)
        width = int(w * r)
        dim = (width, int(height))
        # resize image
        rescaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        # csak a bal felso resze kell a kepnek!
        resized = rescaled[:rescaled.shape[0] // 4, :rescaled.shape[1] // 4]

        #result = cv2.matchTemplate(image_copy, template, cv2.TM_SQDIFF_NORMED)
        result = match_template(resized, template)  #### cv2.matchTemplate func. TM_SQDIFF_NORMED
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print(scale, min_val)
        if(min_val < minVal):
            minVal = min_val
            results = (resized, result)
    return results
# TODO
def matchCharacter(image, template):
    #image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]
    config = '--oem 3  --psm 10 -c tessedit_char_whitelist=AJKQ2345678910'

    #image = cv2.GaussianBlur(image, (1,1), 0)
    # TODO
    # lehetne egy loopot csinalni, 6-8ig levagni a heightet ezalatt
    # es remelhetoleg legfeljebb egyfajta eredmenyt adnak vissza, nem tobbfajtat
    resized = image[:image.shape[0] // 8, :image.shape[1] // 4]
    string = pytesseract.image_to_string(resized, config=config)
    cv2.imshow('matchCharacter',resized)
    return string

def main():
    # read image
    image = cv2.imread('input/main_image_2.jpg')
    # rotate image to x degrees
    image = ndimage.rotate(image, 0)

    # Read the template in grayscale format.
    template = cv2.imread('input/newtemplate1.jpg', 0)
    w, h = template.shape[::-1]

    cv2.imshow('Original', image)

    # convert the image to grayscale
    image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # scale the image and choose the best matching one
    resizedImage, result = matchSymbol(image_copy, template)
    #match character
    print(matchCharacter(image_copy, template))

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(min_val)
    treshold = 0.15
    if(min_val <= treshold):
        print("Smaller than treshold, ACCEPT")
    else:
        print("Larger than treshold, REFUSE")

    # Top left x and y coordinates.
    x1, y1 = min_loc # SQDIFF-nel minLoc, mashol maxLoc
    ## Bottom right x and y coordinates.
    x2, y2 = (x1 + w, y1 + h)
    cv2.rectangle(resizedImage, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('Resized image', resizedImage)



    cv2.imshow('Template', template)
    ## Normalize the result for proper grayscale output visualization.
    cv2.normalize(resizedImage, result, 0, 1, cv2.NORM_MINMAX, -1)

    cv2.imshow('Detected point', result)
    cv2.waitKey(0)
    cv2.imwrite('outputs/image_result.jpg', resizedImage)
    cv2.imwrite('outputs/template_result.jpg', result * 255.)

main()
