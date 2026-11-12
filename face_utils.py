"""
Face Recognition and Liveness Detection utilities for SecureVote.

Engine      : OpenCV YuNet (threshold=0.1) with automatic Haar Cascade fallback.
              Multiple lighting & orientation variants tried: original, flip, CLAHE, gamma.
Recognition : OpenCV SFace — 128-d embedding. Uses alignCrop (5 facial landmarks)
              with automatic robust padded manual crop fallback.
Liveness    : Face-centre spatial variance across short video frames.
Hardware    : Fully optimized for laptop webcams and phone front-facing cameras.

No MediaPipe dependency.
"""

import base64
import io
import os
import pickle
import threading
import urllib.request

import cv2
import numpy as np
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
FACE_MATCH_TOLERANCE = 1.128   # kept for backward compat
EAR_THRESHOLD        = 0.21
BLINK_CONSEC_FRAMES  = 2
_MATCH_THRESHOLD     = 0.363   # SFace cosine-similarity — verification
_DUP_THRESHOLD       = 0.363   # SFace cosine-similarity — duplicate check

# ── Model paths ───────────────────────────────────────────────────────────────
YUNET_PATH = os.path.join(os.path.dirname(__file__), "face_detection_yunet.onnx")
SFACE_PATH = os.path.join(os.path.dirname(__file__), "face_recognition_sface.onnx")

# ── Thread lock — cv2.FaceDetectorYN, CascadeClassifier & SFace are NOT thread-safe
_det_lock = threading.Lock()

# ── YuNet detector (primary) ──────────────────────────────────────────────────
_yunet: "cv2.FaceDetectorYN | None" = None
try:
    if os.path.exists(YUNET_PATH):
        _yunet = cv2.FaceDetectorYN.create(
            YUNET_PATH, "", (320, 320),
            0.1,   # scoreThreshold (permissive for laptop/phone camera lighting)
            0.3,   # nmsThreshold
            5000   # topK
        )
        print("[face_utils] YuNet detector loaded (threshold=0.1).")
    else:
        print("[face_utils] WARNING: face_detection_yunet.onnx not found.")
except Exception as _e:
    print(f"[face_utils] YuNet load failed ({_e}) — using Haar Cascade fallback.")

# ── Haar Cascade (fallback) ───────────────────────────────────────────────────
_haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_haar: "cv2.CascadeClassifier" = cv2.CascadeClassifier(_haar_xml)
if _haar.empty():
    print("[face_utils] WARNING: Haar Cascade could not be loaded.")
else:
    print("[face_utils] Haar Cascade fallback ready.")

# ── SFace recognizer (128-d face embedding) ───────────────────────────────────
recognizer: "cv2.FaceRecognizerSF | None" = None
try:
    if os.path.exists(SFACE_PATH):
        recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
        print("[face_utils] SFace recognizer loaded successfully.")
    else:
        print("[face_utils] WARNING: face_recognition_sface.onnx not found.")
except Exception as _e:
    print(f"[face_utils] SFace recognizer load failed: {_e}")

# Legacy alias
detector = _yunet


# ─────────────────────────────────────────────────────────────────────────────
# Image decoding & normalization
# ─────────────────────────────────────────────────────────────────────────────

def decode_base64_image(b64: str):
    """Base64 data-URL or raw string → uint8 RGB numpy array (or None)."""
    try:
        if not b64:
            return None
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(image, dtype=np.uint8)
    except Exception as exc:
        print(f"[face_utils] decode_base64_image error: {exc}")
        return None


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    """Ensure array is uint8 RGB H×W×3."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr.copy()


def _make_variants(rgb: np.ndarray):
    """
    Yield BGR image variants to handle diverse cameras (laptop/mobile)
    and varying lighting conditions (backlit, dim, low-contrast).
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 1. Original BGR
    yield bgr

    # 2. Horizontal flip (handles mirrored front-camera outputs)
    yield cv2.flip(bgr, 1)

    # 3. CLAHE contrast enhancement on LAB Lightness (helps dim rooms)
    try:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        clahe_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        yield clahe_bgr
    except Exception:
        pass

    # 4. Gamma brightened (lifts dark shadows)
    try:
        lut = np.array([min(255, int(((i / 255.0) ** 0.6) * 255)) for i in range(256)], dtype=np.uint8)
        yield cv2.LUT(bgr, lut)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _yunet_detect(bgr: np.ndarray):
    """Run YuNet on a BGR image. Returns highest-scoring face row (15 floats) or None."""
    if _yunet is None:
        return None
    try:
        h, w = bgr.shape[:2]
        with _det_lock:
            _yunet.setInputSize((w, h))
            _, faces = _yunet.detect(bgr)
        if faces is not None and len(faces) > 0:
            return max(faces, key=lambda f: float(f[14]))
    except Exception as exc:
        print(f"[face_utils] _yunet_detect: {exc}")
    return None


def _haar_to_yunet_row(x: int, y: int, w: int, h: int) -> np.ndarray:
    """Convert Haar Cascade bounding box to 15-float YuNet format."""
    return np.array([
        float(x), float(y), float(w), float(h),
        float(x + w * 0.30), float(y + h * 0.37),  # right eye
        float(x + w * 0.70), float(y + h * 0.37),  # left eye
        float(x + w * 0.50), float(y + h * 0.60),  # nose tip
        float(x + w * 0.37), float(y + h * 0.78),  # right mouth corner
        float(x + w * 0.63), float(y + h * 0.78),  # left mouth corner
        0.80                                       # confidence score
    ], dtype=np.float32)


def _haar_detect(bgr: np.ndarray):
    """Run Haar Cascade fallback on histogram-equalised grayscale."""
    if _haar.empty():
        return None
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        with _det_lock:
            faces = _haar.detectMultiScale(
                gray_eq,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
        return _haar_to_yunet_row(int(x), int(y), int(w), int(h))
    except Exception as exc:
        print(f"[face_utils] _haar_detect: {exc}")
    return None


def detect_face_core(image_array: np.ndarray):
    """
    Detect the most prominent human face across camera variants.
    Returns (bgr_image, face_row) or (None, None).
    """
    if image_array is None:
        return None, None

    rgb = _to_rgb(image_array)

    for bgr in _make_variants(rgb):
        # 1. Primary: YuNet
        face = _yunet_detect(bgr)
        if face is not None:
            return bgr, face

        # 2. Fallback: Haar Cascade
        face = _haar_detect(bgr)
        if face is not None:
            return bgr, face

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_face_encoding(image_array: np.ndarray):
    """
    Detect a face and extract a 128-d SFace embedding.
    Works reliably on laptop webcams and phone front-facing cameras.
    """
    if recognizer is None:
        print("[face_utils] SFace recognizer not loaded.")
        return None

    bgr_img, face_row = detect_face_core(image_array)
    if face_row is None:
        print("[face_utils] get_face_encoding: no face detected in image.")
        return None

    with _det_lock:
        face_row_f32 = np.ascontiguousarray(face_row, dtype=np.float32).reshape(-1)

        # 1️⃣ Try SFace alignCrop (using facial landmarks)
        try:
            aligned = recognizer.alignCrop(bgr_img, face_row_f32)
            if aligned is not None and aligned.size > 0:
                feat = recognizer.feature(aligned)
                if feat is not None:
                    return feat
        except Exception as exc:
            print(f"[face_utils] alignCrop fallback: {exc}")

        # 2️⃣ Robust padded manual crop fallback (resized to 112×112)
        try:
            x, y, w, h = [max(0, int(v)) for v in face_row[:4]]
            pad_w = int(w * 0.15)
            pad_h = int(h * 0.15)
            img_h, img_w = bgr_img.shape[:2]
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(img_w, x + w + pad_w)
            y2 = min(img_h, y + h + pad_h)
            crop = bgr_img[y1:y2, x1:x2]
            if crop.size > 0:
                resized_crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
                resized_crop = np.ascontiguousarray(resized_crop, dtype=np.uint8)
                feat = recognizer.feature(resized_crop)
                if feat is not None:
                    return feat
        except Exception as exc2:
            print(f"[face_utils] manual crop feature extraction failed: {exc2}")

    return None


def detect_face_in_frame(image_array: np.ndarray):
    """
    Quick human presence check for the live camera polling badge.
    Returns (face_detected: bool, center: tuple | None).
    """
    _, face_row = detect_face_core(image_array)
    if face_row is None:
        return False, None
    x, y, w, h = face_row[:4]
    return True, (float(x + w / 2), float(y + h / 2))


def compare_faces(known_encoding, test_image_array: np.ndarray):
    """
    Compare a stored face encoding against a live captured image.
    Returns (match: bool, pseudo_distance: float).
    """
    if recognizer is None:
        return False, 1.0
    try:
        test_enc = get_face_encoding(test_image_array)
        if test_enc is None:
            return False, 1.0
        with _det_lock:
            sim = float(recognizer.match(known_encoding, test_enc, 0))
        dist = 1.0 - max(0.0, sim)
        return sim >= _MATCH_THRESHOLD, dist
    except Exception as exc:
        print(f"[face_utils] compare_faces error: {exc}")
        return False, 1.0


def find_duplicate_face(new_encoding, registered_voters):
    """
    Check whether a new registrant's face matches any already-registered voter.
    Returns (is_duplicate: bool, matched_name: str | None).
    """
    if recognizer is None or new_encoding is None:
        return False, None
    for voter_name, enc_bytes in registered_voters:
        if not enc_bytes:
            continue
        try:
            with _det_lock:
                sim = float(recognizer.match(new_encoding, deserialize_encoding(enc_bytes), 0))
            if sim >= _DUP_THRESHOLD:
                print(f"[face_utils] DUPLICATE face — matches {voter_name!r} (sim={sim:.4f})")
                return True, voter_name
        except Exception as exc:
            print(f"[face_utils] find_duplicate_face error for {voter_name!r}: {exc}")
    return False, None


def save_face_image(image_array: np.ndarray, filepath: str) -> bool:
    """Save RGB face array to disk as JPEG."""
    try:
        Image.fromarray(image_array).save(filepath, "JPEG", quality=90)
        return True
    except Exception as exc:
        print(f"[face_utils] save_face_image error: {exc}")
        return False


def serialize_encoding(encoding) -> bytes:
    """Pickle-serialize an SFace embedding for database storage."""
    return pickle.dumps(encoding)


def deserialize_encoding(enc_bytes: bytes):
    """Pickle-deserialize an SFace embedding from database bytes."""
    return pickle.loads(enc_bytes)


def analyze_liveness_frames(frames_data: list):
    """
    Confirm liveness from a sequence of base64-encoded frames.
    Returns (is_live: bool, score: int, message: str).
    """
    if not frames_data or len(frames_data) < 5:
        return False, 0, "Insufficient frames for liveness detection."

    centers, hits = [], 0
    for fb64 in frames_data:
        img = decode_base64_image(fb64)
        if img is None:
            continue
        ok, center = detect_face_in_frame(img)
        if ok:
            hits += 1
            centers.append(center)

    if hits < len(frames_data) * 0.4:
        return False, 0, "Face not consistently detected. Please look directly into the camera."

    if len(centers) >= 2:
        var_x = float(np.var([c[0] for c in centers]))
        var_y = float(np.var([c[1] for c in centers]))
        if var_x > 0.4 or var_y > 0.4:
            return True, 1, "Liveness confirmed — natural movement detected."

    return False, 0, "No natural movement detected. Please move slightly while facing the camera."
