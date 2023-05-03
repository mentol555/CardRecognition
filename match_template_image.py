import string
import warnings
import cv2
import numpy as np
import pytesseract
from numpy import array

# from scipy import ndimage

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

warnings.filterwarnings('ignore')


# opencv matchTemplate() metódus implementációja
# TM_SQDIFF_NORMED metódus változata
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
                    szamlalo += (template[x, y] - image[i + x, j + y]) ** 2
                    nevezo_1 += template[x, y] ** 2
                    nevezo_2 += image[i + x, j + y] ** 2
            result[i, j] = szamlalo / (np.sqrt(nevezo_1 * nevezo_2))
            if (result[i, j] > 1.0): result[i, j] = 1.0
    return result


# multi scale symbol matching
# 3 méretarányt kipróbál, minta szimbólumot / template-t illeszt,
# a legjobban illeszkedőt választja

def matchSymbol(image, color):
    minVal = 100
    if(color == "red"):
        templates = ["karo.jpg", "kor.jpg"]
    else:
        templates = ["treff.jpg", "pikk.jpg"]

    print("Trying different scales of the image. ")
    print("The lower match value is the better")
    for templ in templates:
        template = cv2.imread('input/'+templ, 0)
        print("Trying template "+templ)

        for scale in (12, 14, 17):
            # rescale, keep the ratio!
            (h, w) = image.shape[:2]
            height = template.shape[0] * scale
            r = height / float(h)
            width = int(w * r)
            dim = (width, int(height))
            # resize image
            rescaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            # csak a bal felso resze kell a kepnek!
            resized = rescaled[:int(image.shape[0] / 4), :int(image.shape[1] / 4)]

            # result = cv2.matchTemplate(image_copy, template, cv2.TM_SQDIFF_NORMED)
            result = match_template(resized, template)  #### cv2.matchTemplate func. TM_SQDIFF_NORMED
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            print("Image scale:", scale, "Match value:", min_val)
            if (min_val < minVal):
                template_name = templ
                minVal = min_val
                results = (resized, result, template, template_name)
    print("Best match with: ", template_name, " value: ", minVal)
    return results




# match character
# visszaadja a felismert karaktert (ha van), mely csakis a whitelist-ből származhat
# pytesseract használata - OCR (Optical Character Recognition)
def matchCharacter(image, template):
    config = '--oem 3  --psm 10 -c tessedit_char_whitelist=AJKQ0123456789Il'
    # image = cv2.GaussianBlur(image, (1,1), 0)
    resized = image[:int(image.shape[0] / 1.5), :int(image.shape[1] / 1.5)]
    string = pytesseract.image_to_string(resized, config=config)
    cv2.imshow('matchCharacter', resized)

    # bug: 10-es kártyánál csak a 0-át veszi észre
    # De mivel csak a 10-esben van, ezért tudjuk, hogy a 10-es ről van szó
    string = string.strip()
    if (string == "0"):
        string = "10"
    return string


def resizeImage(image, height):
    (h, w) = image.shape[:2]
    r = height / float(h)
    width = int(w * r)
    dim = (width, height)
    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def adjustContrastBrightness(img, contrast, brightness):
    adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    return adjusted


# a kartya bal felso sarkat nezzuk (1/8-adat)
# ha itt talal pirosat, akkor szinte biztos hogy piros
# kulonben feketenek fogja elkonyvelni
def checkColor(img):
    resized = resizeImage(img, 500)
    cropped = resized[:int(resized.shape[0] / 8), :int(resized.shape[1] / 8)]
    img_hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    ## Gen lower mask (0-5) and upper mask (175-180) of RED
    mask1 = cv2.inRange(img_hsv, (0, 50, 20), (5, 255, 255))
    mask2 = cv2.inRange(img_hsv, (175, 50, 20), (180, 255, 255))

    ## Merge the mask and crop the red regions
    mask = cv2.bitwise_or(mask1, mask2)
    red_regions = cv2.bitwise_and(cropped, cropped, mask=mask)

    ## Display
    if (len(set(red_regions.flatten())) > 5):
        color = "red"
    else:
        color = "black"
    print("The card is probably ", color)
    return color


def main():
    # read image
    original = cv2.imread('input/img12b.jpg')

    # resize image, height to be 500
    # szelesseg is aranyosan valtozik
    original = resizeImage(original, 500)
    cv2.imshow('Original', original)
    # adjust contrast, brightness
    image = original.copy()
    image = adjustContrastBrightness(image, 1.5, 5)
    cv2.imshow('Adjusted contrast, brightness', image)

    # Kartya szinenek meghatarozasa: piros / fekete
    # A szintol fuggoen ketto eset: karo/kor, treff/pikk

    color = checkColor(original)
    image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resizedImage, result, template, template_name = matchSymbol(image_copy, color)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    w, h = template.shape[::-1]
    # Bal felso x, y koordinatak
    x1, y1 = min_loc  # SQDIFF-nel minLoc, mashol maxLoc
    # Jobb also x, y koordinatak
    x2, y2 = (x1 + w, y1 + h)  # w, h -> template width, height


    ### match character
    # lefedjük fehérrel a részt ahol a szimbólum van, igy nem hat a karakterfelismerésre
    onlyCharacter = resizedImage.copy()
    # lefedjük egy fehér telitett téglalappal
    cv2.rectangle(onlyCharacter, (x1, y1), (x2, y2), (255, 255, 255), -1)
    character = matchCharacter(onlyCharacter, template)
    # ha meg tul sotet lenne a kep a felismereshez
    if(character == ""):
        print("Adjusting contrast and brightness even more to find character")
        onlyCharacter = adjustContrastBrightness(onlyCharacter, 1.5, 5)
        character = matchCharacter(onlyCharacter, template)
    print("Recognised character is: ",character)

    # Vegso eredmeny kiiratasa
    template_name = template_name.replace(".jpg", "")
    print("Result: ", template_name, character)

    # Detektalas kirajzolasa
    # bal felso es jobb felso x, y koordinatak, keret szine, vastagsaga
    cv2.rectangle(resizedImage, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('Symbol location', resizedImage)

    cv2.imshow('Template', template)
    # Normalizalas
    #cv2.normalize(resizedImage, result, 0, 1, cv2.NORM_MINMAX, -1)

    #cv2.imshow('Detected point', result)
    cv2.waitKey(0)

main()
