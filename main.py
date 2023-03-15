import string

import cv2
from scipy import ndimage

def main(img_source: string, template_source: string, deg: int):
    image = cv2.imread('input/main_image_2.jpg')
    image = ndimage.rotate(image, deg)
    image_copy = image.copy()
    # Convert copy of image to Grayscale format as we need that
    # for template matching.
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    # Read the template in grayscale format.
    template = cv2.imread('input/template2.jpg', 0)
    w, h = template.shape[::-1]

    # Apply template Matching.
    result = cv2.matchTemplate(image_copy, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(deg)
    print(max_val)

    # Top left x and y coordinates.
    x1, y1 = max_loc
    ## Bottom right x and y coordinates.
    x2, y2 = (x1 + w, y1 + h)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('Result', image)
    ## Normalize the result for proper grayscale output visualization.
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
    cv2.imshow('Detected point', result)
    cv2.waitKey(0)
    cv2.imwrite('outputs/image_result.jpg', image)
    cv2.imwrite('outputs/template_result.jpg', result * 255.)


print(main("input/main_image.png", "input/template2.jpg", 180))
print(main("input/main_image.png", "input/template2.jpg", 0))
#for i in range(0, 365, 5):
  #  main("input/main_image.png", "input/template2.jpg", i)
