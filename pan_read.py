from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
import cv2
import os
import re
import pdf2image
ocr = PaddleOCR(lang = 'en')

class pan_extract:

    def __init__(self, file_path):
        self.file_path = file_path

    def initialize_ocr(self):
        """Initialize the PaddleOCR instance."""
        return PaddleOCR(use_angle_cls=True, lang='en')
    
    @staticmethod
    def pan_correction(pan):
        pan_new = pan[0][0:5].replace(r'0', 'O') + pan[0][5:9] + pan[0][9].replace('0', 'O')
        return pan_new

    def extract_pan_numbers(self):
        filename = self.file_path
        """Extract PAN numbers from drivers license images in the specified directory."""
        ocr = self.initialize_ocr()
        filtered_numbers = []
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            result = ocr.ocr(filename)
            print(result)
            for res in result:
                for line in res:
                    text = line[1][0]
                    matches = re.findall("^[A-Z0-9]{5}[0-9]{4}[A-Z0-9]{1}", text)
                    if matches:
                        filtered_numbers.extend(matches)
                        filtered_numbers = list(dict.fromkeys(filtered_numbers))
        else:
            pages = pdf2image.convert_from_path(filename, 500)
            for i in pages:
                i.save(filename.split('/')[-1].split('.')[0]+".jpg", 'JPEG')
                result = ocr.ocr(filename.split('/')[-1].split('.')[0]+".jpg")
                print(result)
                for res in result:
                    for line in res:
                        text = line[1][0]
                        matches = re.findall("^[A-Z0-9]{5}[0-9]{4}[A-Z0-9]{1}", text)
                        if matches:
                            filtered_numbers.extend(matches)
                            filtered_numbers = list(dict.fromkeys(filtered_numbers))
        if len(filtered_numbers) == 0:
            return print('No PAN Number found')
        else:
            return self.pan_correction(filtered_numbers)

if __name__ == "__main__":
    # Specify the path to the directory containing voter ID card images
    file_path = r"<enter path to directory containing PAN card images>"

    # Call the function to extract Driver's License numbers
    pan_extract(file_path).extract_pan_numbers()
