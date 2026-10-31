"""
Face Recognition and Liveness Detection utilities for SecureVote.
Uses face_recognition library for face matching and
MediaPipe Face Mesh for blink-based liveness detection.
"""

import base64
import io
import os
import pickle

import cv2
import numpy as np
from PIL import Image

# Thresholds
FACE_MATCH_TOLERANCE = 1.128  # SFace L2 distance threshold
EAR_THRESHOLD = 0.21
BLINK_CONSEC_FRAMES = 2

def _resolve_model_path(filename):
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        os.path.join(os.getcwd(), filename),
        os.path.join("/var/task", filename),
        os.path.join(os.path.dirname(__file__), filename)
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


# OpenCV Face Recognition Models — bundled with the project (no download needed)
YUNET_PATH = _resolve_model_path("face_detection_yunet.onnx")
SFACE_PATH = _resolve_model_path("face_recognition_sface.onnx")

try:
    if os.path.exists(YUNET_PATH) and os.path.exists(SFACE_PATH):
        detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320))
        recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
    else:
        print("Warning: Face model files not found. Face recognition will be disabled.")
        detector = None
        recognizer = None
except Exception as e:
    print(f"Error loading OpenCV Face models: {e}")
    detector = None
    recognizer = None



def decode_base64_image(base64_string):
    """Decode a base64-encoded image string to a numpy array."""
    try:
        # Remove data URL prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return np.array(image)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def get_face_encoding(image_array):
    """
    Extract face encoding from an image array.
    Returns the encoding or None if no face is found.
    """
    if detector is None or recognizer is None:
        print("Face models are not loaded.")
        return None

    try:
        # Convert RGB (from PIL) to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = image_array

        h, w = bgr_image.shape[:2]
        detector.setInputSize((w, h))
        
        status, faces = detector.detect(bgr_image)
        if faces is None or len(faces) == 0:
            return None
            
        # Get the first face
        face = faces[0]
        
        # Align face and extract features
        aligned_face = recognizer.alignCrop(bgr_image, face)
        feature = recognizer.feature(aligned_face)
        return feature
    except Exception as e:
        print(f"Error getting face encoding: {e}")
        return None


def compare_faces(known_encoding, test_image_array):
    """
    Compare a known face encoding with a test image.
    Returns (match: bool, distance: float).
    """
    try:
        test_encoding = get_face_encoding(test_image_array)
        if test_encoding is None:
            return False, 1.0

        # Calculate Cosine Similarity (higher is better, threshold >= 0.363)
        # Using 0 (cv2.FaceRecognizerSF_FR_COSINE) to compute cosine similarity
        similarity = recognizer.match(known_encoding, test_encoding, 0)
        match = similarity >= 0.363
        
        # Convert similarity (-1 to 1) to distance (0 to 1) for the app.py logic
        # app.py uses: confidence = max(0, (1 - distance) * 100)
        # So we want distance to be small when similarity is high.
        # Let's return a pseudo-distance: 1.0 - max(0, similarity)
        pseudo_distance = 1.0 - max(0.0, float(similarity))
        
        return match, pseudo_distance
    except Exception as e:
        print(f"Error comparing faces: {e}")
        return False, 1.0


def find_duplicate_face(new_encoding, registered_voters):
    """
    Check whether a newly captured face matches any already-registered voter.

    Args:
        new_encoding:       The SFace feature vector of the new registrant.
        registered_voters:  List of (voter_name, face_encoding_bytes) tuples
                            fetched from the database.

    Returns:
        (is_duplicate: bool, matched_name: str | None)
        is_duplicate is True if the face is already registered under a different account.
    """
    if recognizer is None or new_encoding is None:
        return False, None

    # SFace cosine similarity threshold — same as login verification
    DUPLICATE_THRESHOLD = 0.363

    for voter_name, encoding_bytes in registered_voters:
        if not encoding_bytes:
            continue
        try:
            stored_encoding = deserialize_encoding(encoding_bytes)
            similarity = recognizer.match(new_encoding, stored_encoding, 0)  # 0 = COSINE
            if similarity >= DUPLICATE_THRESHOLD:
                print(f"[DUPLICATE FACE] Incoming face matches registered voter: {voter_name} (similarity={similarity:.4f})")
                return True, voter_name
        except Exception as e:
            print(f"Error comparing face against voter '{voter_name}': {e}")
            continue

    return False, None



def save_face_image(image_array, filepath):
    """Save a face image to disk."""
    try:
        img = Image.fromarray(image_array)
        img.save(filepath, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"Error saving face image: {e}")
        return False


def serialize_encoding(encoding):
    """Serialize a face encoding to bytes for database storage."""
    return pickle.dumps(encoding)


def deserialize_encoding(encoding_bytes):
    """Deserialize a face encoding from bytes."""
    return pickle.loads(encoding_bytes)


def detect_face_in_frame(image_array):
    """
    Detect face and return its center using OpenCV YuNet.
    Returns (face_detected: bool, center: tuple).
    """
    if detector is None:
        return False, None
    try:
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = image_array

        h, w = bgr_image.shape[:2]
        detector.setInputSize((w, h))
        status, faces = detector.detect(bgr_image)
        
        if faces is None or len(faces) == 0:
            return False, None

        # faces[0] -> [x, y, w, h, ...]
        face = faces[0]
        x, y, w, h = face[:4]
        center = (x + w/2.0, y + h/2.0)
        return True, center
    except Exception as e:
        print(f"Error in face detection: {e}")
        return False, None


def analyze_liveness_frames(frames_data):
    """
    Analyze multiple frames to detect natural movement (liveness).
    frames_data: list of base64-encoded image strings.
    Returns (is_live: bool, movement_score: int, message: str).
    """
    if not frames_data or len(frames_data) < 5:
        return False, 0, "Insufficient frames for liveness detection"

    centers = []
    face_detected_count = 0

    for frame_b64 in frames_data:
        image = decode_base64_image(frame_b64)
        if image is None:
            continue

        detected, center = detect_face_in_frame(image)
        if detected:
            face_detected_count += 1
            centers.append(center)

    if face_detected_count < len(frames_data) * 0.5:
        return False, 0, "Face not consistently detected. Please face the camera directly."

    # Check for natural movement (variance in face center position)
    if len(centers) >= 2:
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        var_x = np.var(xs)
        var_y = np.var(ys)
        
        # If there's slight natural movement, confirm liveness
        if var_x > 0.5 or var_y > 0.5:
            return True, 1, "Liveness confirmed! Natural movement detected."

    return False, 0, "No natural movement detected. Please move slightly while facing the camera."
