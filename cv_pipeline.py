import cv2
import numpy as np
import easyocr
import exifread
from mtcnn import MTCNN
import streamlit as st


@st.cache_resource
def load_models():
    reader = easyocr.Reader(['en'])
    detector = MTCNN()
    return reader, detector


reader, face_detector = load_models()


def blur_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score


def error_level_analysis(image):

    temp_path = "temp_compressed.jpg"
    cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    recompressed = cv2.imread(temp_path)
    diff = cv2.absdiff(image, recompressed)

    ela_score = np.mean(diff)

    return ela_score


def detect_faces(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(rgb)

    return len(faces)

def extract_text_with_boxes(image):

    results = reader.readtext(image)

    texts = []
    boxes = []

    for r in results:
        bbox, text, conf = r
        texts.append(text)
        boxes.append(bbox)

    return texts, boxes


def detect_faces_with_boxes(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(rgb)

    boxes = []

    for f in faces:
        x, y, w, h = f["box"]
        boxes.append((x, y, w, h))

    return len(faces), boxes


def extract_text(image):

    results = reader.readtext(image)
    texts = [r[1] for r in results]

    return texts


def read_metadata(uploaded_file):

    metadata = {}

    try:
        tags = exifread.process_file(uploaded_file)
        for tag in tags:
            metadata[tag] = str(tags[tag])
    except:
        pass

    return metadata


def run_cv_pipeline(image, file):

    blur = blur_detection(image)
    ela = error_level_analysis(image)

    face_count, face_boxes = detect_faces_with_boxes(image)

    texts, text_boxes = extract_text_with_boxes(image)

    metadata = read_metadata(file)

    return {
        "blur_score": float(blur),
        "ela_score": float(ela),
        "faces_detected": face_count,
        "face_boxes": face_boxes,
        "ocr_text": texts,
        "ocr_boxes": text_boxes,
        "metadata": metadata
    }