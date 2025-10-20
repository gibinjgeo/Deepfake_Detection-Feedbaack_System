import cv2
import numpy as np
import mediapipe as mp

def crop_to_224(image, bbox_xywh, margin=0.1):
    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox_xywh

    # Add a small margin to capture more of the face
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
    mp_fd = mp.solutions.face_detection
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_fd.FaceDetection(model_selection=model_selection,
                             min_detection_confidence=min_conf) as detector:
        result = detector.process(image_rgb)

    if not result.detections:
        return None

    H, W = image_bgr.shape[:2]
    bboxes = []
    for det in result.detections:
        rel = det.location_data.relative_bounding_box
        x = int(rel.xmin * W)
        y = int(rel.ymin * H)
        w = int(rel.width * W)
        h = int(rel.height * H)
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(W - x, w))
        h = max(1, min(H - y, h))
        bboxes.append((x, y, w, h))

    # pick the largest face
    largest = max(bboxes, key=lambda b: b[2] * b[3])
    return largest

if __name__ == "__main__":
    # ðŸ”¸ Take input directly from the user
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
            print(f"âœ… Face detected and cropped to 224x224. Saved at {output_path}")
            # Optional: show image in a window
            cv2.imshow("Cropped Face", face_224)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("âŒ Cropping failed due to invalid bounding box.")
