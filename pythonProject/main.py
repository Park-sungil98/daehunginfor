
from PIL import Image
from pytesseract import *


def orc_test():
    filename = 'ad_text1.jpg'
    image = Image.open(filename)
    text = image_to_string(image, lang='kor')

    print("================ OCR result ================")
    print(text)

orc_test()