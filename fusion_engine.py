def compute_risk_score(cv_results, vision_results):

    score = 0
    indicators = []

    # CV Signals
    if cv_results["blur_score"] < 50:
        score += 10
        indicators.append("Image appears blurry")

    if cv_results["ela_score"] > 15:
        score += 25
        indicators.append("Compression anomaly detected")

    if cv_results["faces_detected"] == 0:
        score += 30
        indicators.append("No face detected")

    if "Photoshop" in str(cv_results["metadata"]):
        score += 30
        indicators.append("Editing software detected in metadata")

    # Vision Signals
    if vision_results.get("layout_consistency") == "suspicious":
        score += 25
        indicators.append("Layout inconsistencies detected")

    if vision_results.get("photo_region") == "suspicious":
        score += 35
        indicators.append("Possible photo replacement")

    if vision_results.get("risk_level") == "high":
        score += 40
        indicators.append("Vision model flagged high risk")

    # Risk Level
    if score < 30:
        level = "Likely Genuine"
    elif score < 60:
        level = "Moderate Risk"
    else:
        level = "Suspicious"

    return score, level, indicators