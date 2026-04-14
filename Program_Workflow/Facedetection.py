import urllib.request
import cv2
import mediapipe as mp
from pathlib import Path

# mediapipe 0.10+ removed mp.solutions in favour of the Tasks API.
# The Tasks API requires a small TFLite model file (~224 KB).
_MODEL_FILENAME = "blaze_face_short_range.tflite"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)


def _get_model_path() -> str:
    """Return path to the TFLite model, downloading it if absent."""
    # Look next to this file so the model travels with the code
    model_path = Path(__file__).parent / _MODEL_FILENAME
    if not model_path.exists():
        print(f"[FaceDetector] Downloading model -> {model_path} ...")
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        print("[FaceDetector] Download complete.")
    return str(model_path)


class FaceDetector:
    """
    Reusable MediaPipe face detector (Tasks API, mediapipe >= 0.10).
    Create once, call detect_largest_face() per frame.

    Usage:
        with FaceDetector() as fd:
            for frame in frames:
                bbox = fd.detect_largest_face(frame)   # (x, y, w, h) or None
    """

    def __init__(self, min_conf: float = 0.5, model_selection: int = 1):
        # model_selection is kept for API compatibility but ignored by the
        # Tasks API (it uses a single unified model).
        model_path = _get_model_path()
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_conf,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    def detect_largest_face(self, image_bgr):
        """
        Returns (x, y, w, h) of the largest detected face in pixel coords,
        or None if no face is found.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._detector.detect(mp_image)

        if not result.detections:
            return None

        H, W = image_bgr.shape[:2]
        bboxes = []
        for det in result.detections:
            bb = det.bounding_box
            x = max(0, bb.origin_x)
            y = max(0, bb.origin_y)
            w = max(1, min(W - x, bb.width))
            h = max(1, min(H - y, bb.height))
            bboxes.append((x, y, w, h))

        return max(bboxes, key=lambda b: b[2] * b[3])

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def crop_to_224(image, bbox_xywh, margin=0.1):
    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox_xywh

    pad = int(max(w, h) * margin)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w_img, x + w + pad)
    y1 = min(h_img, y + h + pad)

    if x1 <= x0 or y1 <= y0:
        return None

    face = image[y0:y1, x0:x1].copy()
    face_224 = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
    return face_224


def detect_largest_face_bgr(image_bgr, min_conf=0.5, model_selection=1):
    """
    Convenience one-shot wrapper. Fine for single images.
    For batch/frame processing use FaceDetector directly to avoid
    creating a new mediapipe instance on every call.
    """
    with FaceDetector(min_conf=min_conf, model_selection=model_selection) as fd:
        return fd.detect_largest_face(image_bgr)


if __name__ == "__main__":
    image_path = "img.png"
    output_path = "face_224.jpg"

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image at {image_path}")
        exit()

    bbox = detect_largest_face_bgr(img)
    if bbox is None:
        print("No face detected in the image.")
    else:
        face_224 = crop_to_224(img, bbox)
        if face_224 is not None:
            cv2.imwrite(output_path, face_224)
            print(f"Face detected and cropped to 224x224. Saved at {output_path}")
            cv2.imshow("Cropped Face", face_224)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Cropping failed due to invalid bounding box.")
