import streamlit as st
import numpy as np
import cv2
from PIL import Image

from cv_pipeline import run_cv_pipeline
from fusion_engine import compute_risk_score
from vision_model import analyze_image_with_vision


st.set_page_config(page_title="AI ID Fraud Detector", layout="wide")

st.title("AI ID Document Forgery Detector")

uploaded_file = st.file_uploader(
    "Upload ID Document Image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Draw OCR + Face Detection
# -----------------------------
def draw_visualizations(image, cv_results):

    img = image.copy()

    # OCR boxes
    for box in cv_results.get("ocr_boxes", []):
        pts = np.array(box).astype(int)

        cv2.polylines(
            img,
            [pts],
            True,
            (0,255,0),
            2
        )

    # Face boxes
    for (x,y,w,h) in cv_results.get("face_boxes", []):

        cv2.rectangle(
            img,
            (x,y),
            (x+w, y+h),
            (255,0,0),
            2
        )

    return img


# -----------------------------
# Forgery Heatmap (ELA)
# -----------------------------
def generate_forgery_heatmap(image):

    _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    recompressed = cv2.imdecode(encoded, 1)

    diff = cv2.absdiff(image, recompressed)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    return heatmap


def overlay_heatmap(image, heatmap, alpha=0.6):

    overlay = cv2.addWeighted(
        heatmap,
        alpha,
        image,
        1-alpha,
        0
    )

    return overlay


# -----------------------------
# Main App
# -----------------------------
if uploaded_file:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Save image for vision model API
    image_path = "uploaded_image.jpg"
    image.save(image_path)

    # Show uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Fraud Analysis"):

        # Convert PIL → OpenCV
        cv_image = np.array(image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        # -----------------------------
        # CV Pipeline
        # -----------------------------
        with st.spinner("Running image forensic analysis..."):
            cv_results = run_cv_pipeline(cv_image, uploaded_file)

        # -----------------------------
        # Draw OCR + Face boxes
        # -----------------------------
        vis_image = draw_visualizations(cv_image, cv_results)

        st.write("## Visual Document Analysis")

        st.image(
            cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB),
            caption="OCR (Green) and Face Detection (Blue)",
            use_column_width=True
        )

        # -----------------------------
        # Forgery Heatmap
        # -----------------------------
        heatmap = generate_forgery_heatmap(cv_image)

        overlay = overlay_heatmap(cv_image, heatmap)

        st.write("## Forgery Heatmap Detection")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
                caption="Forgery Heatmap",
                use_column_width=True
            )

        with col2:
            st.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                caption="Heatmap Overlay on Document",
                use_column_width=True
            )

        # -----------------------------
        # Vision Model
        # -----------------------------
        with st.spinner("Running vision model analysis..."):
            vision_results = analyze_image_with_vision(image_path)

        # -----------------------------
        # Fusion Scoring
        # -----------------------------
        score, level, indicators = compute_risk_score(cv_results, vision_results)

        st.subheader("Fraud Detection Report")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Computer Vision Signals")
            st.json(cv_results)

        with col2:
            st.write("### Vision Model Analysis")
            st.json(vision_results)

        st.divider()

        st.write("### Final Risk Score")
        st.metric("Risk Score", score)

        st.write("### Risk Level")

        if level == "Suspicious":
            st.error(level)
        elif level == "Moderate Risk":
            st.warning(level)
        else:
            st.success(level)

        st.write("### Indicators")

        if indicators:
            for i in indicators:
                st.write("-", i)
        else:
            st.write("No suspicious indicators detected.")