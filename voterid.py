from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
import cv2
import os
import re
import pdf2image
ocr = PaddleOCR(lang = 'en')

class epic_extract:

    def __init__(self, file_path):
        self.file_path = file_path

    def initialize_ocr(self):
        """Initialize the PaddleOCR instance."""
        return PaddleOCR(use_angle_cls=True, lang='en')

    def extract_epic_numbers(self):
        filename = self.file_path
        """Extract EPIC numbers from voter ID card images in the specified directory."""
        ocr = self.initialize_ocr()
        filtered_numbers = []
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            result = ocr.ocr(filename)
            for res in result:
                for line in res:
                    text = line[1][0]
                    matches = re.findall("^[A-Za-z]{3}\d{7}$", text)
                    if matches:
                        filtered_numbers.extend(matches)
                        filtered_numbers = list(dict.fromkeys(filtered_numbers))
        else:
            pages = pdf2image.convert_from_path(filename, 300)
            for i in pages:
                i.save(filename.split('/')[-1].split('.')[0]+".jpg", 'JPEG')
                result = ocr.ocr(filename.split('/')[-1].split('.')[0]+".jpg")
                print(result)
                for res in result:
                    for line in res:
                        text = line[1][0]
                        matches = re.findall("^[A-Za-z]{3}\d{7}$", text)
                        if matches:
                            filtered_numbers.extend(matches)
                            filtered_numbers = list(dict.fromkeys(filtered_numbers))
        if len(filtered_numbers) == 0:
            return print('No EPIC No. found')
        else:
            return filtered_numbers

if __name__ == "__main__":
    # Specify the path to the directory containing voter ID card images
    file_path = r"<enter path to directory containing votersid card images>"

    # Call the function to extract EPIC numbers
    epic_extract(file_path, confidence_threshold=0.88).extract_epic_numbers()
