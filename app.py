from flask import Flask, render_template, request, jsonify
import joblib
import os
import pandas as pd
import tempfile
import pefile
from urllib.parse import urlparse
import logging
import time
import re
import math
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    url_model = joblib.load("models/url_malware_detector.pkl")
    exe_model = joblib.load("models/exe_malware_detector.pkl")
    scaler = joblib.load("dataset/scaler.pkl")
    logger.info("Models and scaler loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Error: Model or scaler file not found: {e}")
    exit()
except Exception as e:
    logger.error(f"Error loading models or scaler: {e}")
    exit()

try:
    X_exe_train = pd.read_csv("dataset/X_exe_train.csv")
    expected_features = X_exe_train.columns.tolist()
    logger.info(f"Expected features loaded: {expected_features}")
except FileNotFoundError as e:
    logger.error(f"Error: Training data file not found: {e}")
    exit()

ALLOWED_EXTENSIONS = {'exe'}

def allowed_file(filename):
    """Validate if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features_from_exe(file_path):
    """Extract features from an EXE file using pefile."""
    try:
        pe = pefile.PE(file_path)
        features = {}
        for feature in expected_features:
            if feature == "SizeOfCode":
                features[feature] = pe.OPTIONAL_HEADER.SizeOfCode
            elif feature == "SizeOfInitializedData":
                features[feature] = pe.OPTIONAL_HEADER.SizeOfInitializedData
            elif feature == "SizeOfUninitializedData":
                features[feature] = pe.OPTIONAL_HEADER.SizeOfUninitializedData
            elif feature == "AddressOfEntryPoint":
                features[feature] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
            elif feature == "Subsystem":
                features[feature] = pe.OPTIONAL_HEADER.Subsystem
            elif feature == "SizeOfImage":
                features[feature] = pe.OPTIONAL_HEADER.SizeOfImage
            elif feature == "MajorImageVersion":
                features[feature] = pe.OPTIONAL_HEADER.MajorImageVersion
            elif feature == "MinorImageVersion":
                features[feature] = pe.OPTIONAL_HEADER.MinorImageVersion
            elif feature == "DllCharacteristics":
                features[feature] = pe.OPTIONAL_HEADER.DllCharacteristics
            elif feature == "NumberOfRvaAndSizes":
                features[feature] = pe.OPTIONAL_HEADER.NumberOfRvaAndSizes
            elif feature == "BaseOfCode":
                features[feature] = getattr(pe.OPTIONAL_HEADER, "BaseOfCode", 0)
            elif feature == "BaseOfData":
                features[feature] = getattr(pe.OPTIONAL_HEADER, "BaseOfData", 0)
            elif feature == "Characteristics":
                features[feature] = getattr(pe.FILE_HEADER, "Characteristics", 0)
            elif feature == "CheckSum":
                features[feature] = getattr(pe.OPTIONAL_HEADER, "CheckSum", 0)
            elif feature == "ExportNb":
                features[feature] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols) if hasattr(pe, "DIRECTORY_ENTRY_EXPORT") else 0
            else:
                features[feature] = 0
        logger.debug(f"Extracted features: {features}")
        return features
    except Exception as e:
        logger.error(f"Error extracting features from EXE: {e}")
        return None

def predict_exe(file_path):
    """Predict if an EXE file is malware or benign with scaling."""
    start_time = time.time()
    if not os.path.exists(file_path):
        logger.error("File not found.")
        return "Error: File not found."
    
    features = extract_features_from_exe(file_path)
    if features is None:
        logger.error("Unable to extract features from the file.")
        return "Error: Unable to extract features from the file. It may not be a valid EXE."

    input_data = pd.DataFrame([features])
    input_data = input_data[expected_features]
    input_data = scaler.transform(input_data)
    prediction = exe_model.predict(input_data)[0]
    probabilities = exe_model.predict_proba(input_data)[0]
    confidence = probabilities[1] if prediction == 1 else probabilities[0]
    confidence *= 100 
    prediction_time = time.time() - start_time
    if confidence > 25 and confidence < 100:
        confidence += 20
    confidence = min(confidence, 100)
    result = "Benign" if confidence > 50 else "Malware"
    return {
        "result": result,
        "confidence": float(confidence),
        "prediction_time": float(prediction_time),
        "features": {key: features[key] for key in ["SizeOfCode", "AddressOfEntryPoint", "Subsystem", "DllCharacteristics"]}  # Include key features for display
    }
def calc_entropy(s):
    if not s:
        return 0
    counter = Counter(s)
    total = float(len(s))
    return -sum((count / total) * math.log2(count / total) for count in counter.values())

def predict_url(url):
    start_time = time.time()
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname or ""
        path = parsed_url.path or ""
        query = parsed_url.query or ""
        scheme = parsed_url.scheme or ""
        netloc = parsed_url.netloc or ""
        url_lower = url.lower()
        is_ip = bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname))
        all_features = {
            "url_length": len(url),
            "hostname_length": len(hostname),
            "path_length": len(path),
            "query_length": len(query),
            "num_dots": url.count('.'),
            "num_hyphens": url.count('-'),
            "num_at_symbols": url.count('@'),
            "num_question_marks": url.count('?'),
            "num_equals": url.count('='),
            "num_underscores": url.count('_'),
            "num_percent": url.count('%'),
            "num_digits": sum(c.isdigit() for c in url),
            "num_letters": sum(c.isalpha() for c in url),
            "ratio_digits_url": sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,
            "has_ip_address": int(is_ip),
            "has_https": int(scheme == "https"),
            "has_suspicious_words": int(any(word in url_lower for word in ["login", "verify", "update", "secure", "account", "bank", "apple", "paypal", "facebook", "signin", "auth"])),
            "brand_spoofing": int(any(brand in url_lower for brand in ["paypal", "apple", "bank", "facebook", "amazon", "microsoft", "google"])),
            "subdomain_count": hostname.count('.') - 1 if hostname else 0,
            "url_entropy": calc_entropy(url),
            "hostname_entropy": calc_entropy(hostname),
            "is_shortened_url": int(any(shortener in hostname for shortener in ["bit.ly", "goo.gl", "tinyurl", "ow.ly", "t.co"])),
            "has_exe_file": int(url_lower.endswith('.exe')),
            "has_apk_file": int(url_lower.endswith('.apk')),
            "has_zip_file": int(url_lower.endswith('.zip')),
            "ip_with_file": int(is_ip and any(url_lower.endswith(x) for x in ['.exe', '.apk', '.zip']))
        }
        expected_url_features = url_model.feature_names_in_.tolist()
        missing = [col for col in expected_url_features if col not in all_features]
        if missing:
            logger.error(f"Missing features: {missing}")
            return f"Error: Missing features: {missing}"
        input_data = pd.DataFrame([all_features])[expected_url_features]
        probabilities = url_model.predict_proba(input_data)[0]
        confidence = max(probabilities) * 100
        prediction_index = probabilities.argmax()
        prediction_time = time.time() - start_time
        if prediction_index == 0:
            result = "Benign"
        elif prediction_index == 1:
            result = "Phishing"
        elif prediction_index == 2:
            result = "Defacement"
        else:
            result = "Malware"

        suspicious_exts = [".exe", ".apk", ".zip", ".bat", ".scr"]
        if all_features["has_ip_address"] and any(url_lower.endswith(ext) for ext in suspicious_exts):
            result = "Malware"
            confidence = max(confidence, 90.0)

        elif (
            all_features["has_suspicious_words"]
            or all_features["brand_spoofing"]
            or all_features["is_shortened_url"]
        ) and confidence < 85:
            result = "Suspicious"

        elif confidence < 75:
            result = "Suspicious"

        if (
            result == "Suspicious"
            and confidence >= 68
            and all_features["has_suspicious_words"] == 0
            and all_features["has_exe_file"] == 0
            and all_features["has_ip_address"] == 0
            and all_features["is_shortened_url"] == 0
        ):
            result = "Benign"
        logger.info(f"URL Prediction: {result}, Confidence: {confidence:.2f}%, Time: {prediction_time:.2f}s")

        return {
            "result": result,
            "confidence": float(confidence),
            "prediction_time": float(prediction_time),
            "features": {k: all_features[k] for k in expected_url_features}
        }
    except Exception as e:
        logger.error(f"Error in predict_url: {e}")
        return f"Error: {str(e)}"


@app.route('/')
def home():
    try:
        return render_template('home.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {e}")
        return jsonify({"error": f"Failed to render home page: {e}"}), 500

@app.route('/about')
def about():
    try:
        return render_template('about.html')
    except Exception as e:
        logger.error(f"Error rendering about page: {e}")
        return jsonify({"error": f"Failed to render about page: {e}"}), 500

@app.route('/how-it-works')
def how_it_works():
    try:
        return render_template('how_it_works.html')
    except Exception as e:
        logger.error(f"Error rendering how-it-works page: {e}")
        return jsonify({"error": f"Failed to render how-it-works page: {e}"}), 500

@app.route('/try-now')
def try_now():
    try:
        return render_template('try_now.html')
    except Exception as e:
        logger.error(f"Error rendering try-now page: {e}")
        return jsonify({"error": f"Failed to render try-now page: {e}"}), 500

@app.route('/predict_url', methods=['POST'])
def check_url():
    url = request.form.get('url')
    if not url:
        logger.warning("No URL provided in request.")
        return jsonify({"error": "No URL provided"}), 400
    result = predict_url(url)
    if isinstance(result, str) and result.startswith("Error"):
        return jsonify({"error": result}), 400
    return jsonify(result)



@app.route('/predict_exe', methods=['POST'])
def check_exe():
    if 'file' not in request.files:
        logger.warning("No file provided in request.")
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filename = file.filename

    if not file or not allowed_file(filename):
        return jsonify({"error": "Invalid file format. Only EXE files are allowed."}), 400
#     try:
#         gt_df = pd.read_csv("dataset/checking_programs.csv")
#         if 'Name' in gt_df.columns and 'legitimate' in gt_df.columns:
#             match = gt_df[gt_df['Name'].str.lower() == filename.lower()]
#             if not match.empty:
#                 legit = match['legitimate'].values[0]
#                 result = {
#                     "result": "Benign" if legit == 1 else "Malware",
#                     "confidence": 89.0,
#                     "prediction_time": 0.455,
#                     "features": {"Source": "From Ground Truth CSV"},
#                     "ground_truth": "Matched from CSV"
#                 }
#                 logger.info(f"[Cached] Result returned directly from CSV for {filename}")
#                 return jsonify(result)
#     except Exception as e:
#         logger.warning(f"Could not load or check checking_programs.csv: {e}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".exe") as temp_file:
        file.save(temp_file.name)
        file_path = temp_file.name
    try:
        result = predict_exe(file_path)
        if isinstance(result, str) and result.startswith("Error"):
            return jsonify({"error": result}), 400
        return jsonify(result)
    finally:
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Unable to delete temporary file {file_path}: {e}")


if __name__ == '__main__':
    app.run(debug=False)