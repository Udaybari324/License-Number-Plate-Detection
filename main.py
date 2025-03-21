import cv2
import pytesseract
import matplotlib.pyplot as plt
import re

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_number_plate(image_path):
    print(f"Loading image from: {image_path}")
    
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Unable to read the image.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    
    edged = cv2.Canny(thresh, 30, 200)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    number_plate_contour = None
    for contour in contours:
        epsilon = 0.018 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            number_plate_contour = approx
            break
    
    if number_plate_contour is None:
        print("Error: Number plate contour not found.")
        return None

    try:
        cv2.drawContours(img, [number_plate_contour], -1, (0, 255, 0), 3)
        
        # Crop the number plate
        (x, y, w, h) = cv2.boundingRect(number_plate_contour)
        cropped_number_plate = gray[y:y + h, x:x + w]

        text = pytesseract.image_to_string(cropped_number_plate, config='--psm 8')
        
        # Extracted text (cleaned up)
        text = re.sub(r'\W+', '', text)
        
        # Display or save the modified image with contour
        cv2.imshow('Detected Plate', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return text.strip()

    except Exception as e:
        print(f"Error during OCR: {e}")
        return None

def main():
    # image_path = r'C:\Users\user\Downloads\np1.jpeg'    
    # image_path = r'C:\Users\user\Downloads\np2.jfif'
    # image_path = r'C:\Users\user\Downloads\np3.jpg'
    # image_path = r'C:\Users\user\Downloads\np4.jpg'    
    
    number_plate = detect_number_plate(image_path)
    
    if number_plate:
        print(f"Detected Number Plate: {number_plate}")
    else:
        print("Number plate detection failed.")

if __name__ == "__main__":
    main()

