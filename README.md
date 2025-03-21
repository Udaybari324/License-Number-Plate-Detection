# 🚗 License Number Plate Detection 🔍  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) 
![OpenCV](https://img.shields.io/badge/OpenCV-✔️-green)  
![Tesseract OCR](https://img.shields.io/badge/Tesseract%20OCR-✔️-orange)  

🚀 **License Number Plate Detection** is a computer vision project that detects and extracts license plate numbers from images using **OpenCV** and **Tesseract OCR**.

---

## 📌 Features  
✅ **Preprocessing** - Grayscale conversion, noise reduction, edge detection  
✅ **Contour Detection** - Identifies potential number plates  
✅ **OCR Processing** - Extracts text from the detected plate  
✅ **Interactive Display** - Shows the detected plate and OCR results  

---

## 🛠️ Tech Stack  
🔹 Python  
🔹 OpenCV  
🔹 Tesseract OCR  
🔹 Matplotlib  
🔹 Regular Expressions  

---

## 📸 Sample Output  

| Input Image  | Processed Image | Extracted Text |
|-------------|---------------|----------------|
| ![Input](https://via.placeholder.com/150) | ![Processed](https://via.placeholder.com/150) | **MH12AB1234** |

---

## 🚀 Installation & Setup  

1️⃣ **Clone the Repository**  
```sh
git clone https://github.com/Udaybari324/License-Number-Plate-Detection.git
cd license-plate-detection
```

2️⃣ **Install Dependencies**  
```sh
pip install opencv-python pytesseract matplotlib
```

3️⃣ **Set Up Tesseract OCR**  
- Download & install **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)**
- Set its path in the script:  
  ```python
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```

---

## 🔧 Usage  

1️⃣ Place your image inside the project folder  
2️⃣ Run the script:  
```sh
python detect_plate.py
```
3️⃣ The detected number plate will be displayed along with the extracted text  

---

## 🎯 Future Improvements  
📌 Implement real-time number plate detection using a webcam  
📌 Enhance OCR accuracy with deep learning models  
📌 Integrate with a database for vehicle record management  

---

## 📜 License  
📝 This project is open-source and available under the **MIT License**.  

---

👨‍💻 **Developed by [Your Name](https://github.com/Udaybari324) 🚀**  
⭐ _If you like this project, consider giving it a star!_ 🌟
