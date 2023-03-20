import string
import warnings
import cv2
import numpy as np
from scipy import ndimage

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

def main():
    image = cv2.imread('input/aceheart2.png')
    image = ndimage.rotate(image, 0)
    # Convert copy of image to Grayscale format as we need that
    # for template matching.


    width = int(image.shape[1] * 20 / 100)
    height = int(image.shape[0] * 20 / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image_copy = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Read the template in grayscale format.
    template = cv2.imread('input/template3.jpg', 0)
    width = int(template.shape[1] * 60 / 100)
    height = int(template.shape[0] * 60 / 100)
    dim = (width, height)
    # resize image
    newtemplate = cv2.resize(template, dim, interpolation=cv2.INTER_AREA)
    w, h = newtemplate.shape[::-1]

    # Apply template Matching.
    #testimage = np.full((10, 10), [255, 100, 50, 255, 100, 50, 255, 100, 50, 30], dtype=np.uint8)
    #testtempl = np.full((2, 2), 240, dtype=np.uint8)

    #result = cv2.matchTemplate(image_copy, newtemplate, cv2.TM_SQDIFF_NORMED)
    result = match_template(image_copy, newtemplate)  #### cv2.matchTemplate func. TM_SQDIFF_NORMED

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)


    print(max_val)
    # Top left x and y coordinates.
    x1, y1 = min_loc # SQDIFF-nel minLoc, mashol maxLoc
    ## Bottom right x and y coordinates.
    x2, y2 = (x1 + w, y1 + h)
    cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('Image', resized)
    ## Normalize the result for proper grayscale output visualization.
    cv2.normalize(resized, result, 0, 1, cv2.NORM_MINMAX, -1)
    cv2.imshow('Detected point', result)
    cv2.waitKey(0)
    cv2.imwrite('outputs/image_result.jpg', resized)
    cv2.imwrite('outputs/template_result.jpg', result * 255.)

main()
