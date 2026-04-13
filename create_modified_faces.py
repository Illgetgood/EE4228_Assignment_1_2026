import os
from PIL import Image, ImageOps
from offset_face import CropFace
import cv2
import numpy as np

# Path to dataset (dynamic based on current directory)
BASE_PATH = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

TXT_FILE = os.path.join(BASE_PATH, "dataset", "member_faces", "member.txt")
# Output folder
OUTPUT_FOLDER = os.path.join(BASE_PATH, "dataset", "member_faces", "modified")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -----------------------------
# Load DNN face detector
# -----------------------------
# If files are in the same folder as the script
proto_path = os.path.join(os.getcwd(), "deploy.prototxt.txt")
model_path = os.path.join(os.getcwd(), "res10_300x300_ssd_iter_140000.caffemodel")

face_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# Load facial landmark detector
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")  # Pretrained LBF model from OpenCV

# -----------------------------
# Detect face and eyes function with rotation
# -----------------------------
def detect_face_and_eyes(image_path):
    original_img = cv2.imread(image_path)
    if original_img is None:
        return None, None, None

    pil_image_original = Image.open(image_path)
    pil_image_original = ImageOps.exif_transpose(pil_image_original)  # Sync with CV2 EXIF handling

    rotations = [
        (0, None), 
        (90, cv2.ROTATE_90_CLOCKWISE), 
        (180, cv2.ROTATE_180), 
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    for angle, cv2_rot_flag in rotations:
        img = original_img.copy()
        pil_img = pil_image_original.copy()
        
        if cv2_rot_flag is not None:
            img = cv2.rotate(img, cv2_rot_flag)
            if angle == 90:
                pil_img = pil_img.transpose(Image.ROTATE_270)
            elif angle == 180:
                pil_img = pil_img.transpose(Image.ROTATE_180)
            elif angle == 270:
                pil_img = pil_img.transpose(Image.ROTATE_90)

        # Detect via DNN
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        face_net.setInput(blob)
        detections = face_net.forward()

        best_confidence = 0
        best_box = None
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > best_confidence:
                best_confidence = confidence
                best_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        if best_confidence > 0.5 and best_box is not None:
            (startX, startY, endX, endY) = best_box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)
            face_w, face_h = endX - startX, endY - startY
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_rects = np.array([[startX, startY, face_w, face_h]])
            ok, landmarks = facemark.fit(gray, faces_rects)

            if ok and len(landmarks) > 0:
                points = landmarks[0][0]
                eye_left = (int(points[36][0]), int(points[36][1]))
                eye_right = (int(points[45][0]), int(points[45][1]))
                return pil_img, eye_left, eye_right
            else:
                eye_left = (int(startX + face_w * 0.3), int(startY + face_h * 0.35))
                eye_right = (int(startX + face_w * 0.7), int(startY + face_h * 0.35))
                return pil_img, eye_left, eye_right

    # Fallback if no face is found
    h, w = original_img.shape[:2]
    eye_left = (int(w * 0.35), int(h * 0.4))
    eye_right = (int(w * 0.65), int(h * 0.4))
    return pil_image_original, eye_left, eye_right


def readFileNames():
    try:
        inFile = open(TXT_FILE)
    except:
        raise IOError("Cannot find yale.txt in " + BASE_PATH)

    picPath = []
    picIndex = []

    for line in inFile.readlines():
        if line.strip() != "":
            fields = line.rstrip().split(';')
            picPath.append(fields[0])
            picIndex.append(int(fields[1]))

    return (picPath, picIndex)


def main():
    images, indexes = readFileNames()

    # Create output folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for img in images:
        # Full image path
        img_path = os.path.join(BASE_PATH, img)

        # Extract subject folder and filename
        subject = os.path.basename(os.path.dirname(img_path))   # subject14
        filename = os.path.basename(img_path).split('.')[0]     # sad

        # Output directory
        output_dir = os.path.join(OUTPUT_FOLDER, subject)

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        pil_image, eye_left, eye_right = detect_face_and_eyes(img_path)

        if pil_image is None or eye_left is None:
            print("Could not process image:", img_path)
            continue

        # Generate cropped face
        CropFace(
            pil_image,
            eye_left=eye_left,
            eye_right=eye_right,
            offset_pct=(0.3,0.3),
            dest_sz=(200,200)
        ).save(os.path.join(output_dir, filename + "_30_30_200_200.jpg"))

        print("Processed:", filename)


if __name__ == "__main__":
    main()