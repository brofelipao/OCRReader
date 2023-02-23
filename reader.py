import cv2
import pytesseract
import numpy as np
import re

def filtra_texto(texto):
    texto = re.sub('\W', '', texto)
    return re.sub('\s', '', texto)

config = '--oem 3 --psm 6'
img = cv2.imread('imagens/download.png')
img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel)
erosion = cv2.erode(opening, kernel, iterations = 1)

blurred = cv2.GaussianBlur(erosion, (5, 5), 0)
thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#cv2.imshow('img', thresh)
#cv2.waitKey(0)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\luiz.monteiro\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
result = pytesseract.image_to_string(thresh, config=config)
print(filtra_texto(result))
