# AI ID Document Forgery Detector

This project analyzes uploaded ID document images and produces a fraud risk report.

## Features

- OCR text detection
- Face detection
- Metadata inspection
- Error Level Analysis (ELA)
- Forgery heatmap visualization
- Vision model reasoning
- Risk scoring engine

## Tech Stack

- Streamlit
- OpenCV
- EasyOCR
- MTCNN
- ExifRead
- Groq Vision Model

## Example Workflow

- Upload ID document
      Upload an ID image and the system will:
      - Perform forensic image analysis
      - Detect text and faces
      - Highlight possible tampered regions
      - Generate a fraud risk score and report

- System runs forensic analysis

- Vision model evaluates visual inconsistencies

- Signals are combined into a fraud score

- Results displayed in the dashboard

## Example outputs include:

- OCR bounding boxes
- Face detection
- Forgery heatmap
- Fraud risk report

## Run the App

Install dependencies:
pip install -r requirements.txt

Environment Variables:
(Create a .env file in the project root.)
GROQ_API_KEY=your_groq_api_key

Run the app:

streamlit run app.py
