from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_image(filepath):
    image = Image.open(filepath)
    text = pytesseract.image_to_string(image)

    return text

image_path = '../images/hello_world.png'
text = ocr_image(image_path)
print(text)