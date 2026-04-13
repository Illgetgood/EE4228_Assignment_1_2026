import cv2
import numpy as np
import os
import sys
import math
from collections import deque

def norm_0_255(src):
    """Normalize image to [0, 255] for display."""
    if src.dtype != np.uint8:
        dst = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    else:
        dst = src.copy()
    return dst

def tan_triggs_preprocessing(src, gamma=0.2, sigma0=1.0, sigma1=2.0, sz=11, alpha=0.1, tau=10.0):
    """Tan and Triggs illumination normalization."""
    img = src.astype(np.float32)
    
    # 1. Gamma Correction
    img = np.power(img, gamma)
    
    # 2. Difference of Gaussian (DoG)
    img0 = cv2.GaussianBlur(img, (sz, sz), sigma0)
    img1 = cv2.GaussianBlur(img, (sz, sz), sigma1)
    img = img0 - img1
    
    # 3. Contrast Equalization
    abs_img = np.abs(img)
    abs_img = np.power(abs_img, alpha)
    mean_a = np.mean(abs_img)
    img = img / np.power(mean_a, 1.0 / alpha)
    
    abs_img = np.abs(img)
    abs_img = np.minimum(abs_img, tau)
    abs_img = np.power(abs_img, alpha)
    mean_t = np.mean(abs_img)
    img = img / np.power(mean_t, 1.0 / alpha)
    
    # Final normalization to [0, 255]
    dst = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return dst

def build_label_name_map(csv_file, separator=';'):
    """Read CSV, resize images, populate labels, and build label->name map."""
    images = []
    labels = []
    label_names = {}
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return images, labels, label_names
        
    with open(csv_file, 'r') as f:
        for line in f:
            parts = line.strip().split(separator)
            if len(parts) < 2:
                continue
            path, label_str = parts[0], parts[1]
            label = int(label_str)
            
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Could not read image: {path}")
                continue
                
            img = cv2.resize(img, (200, 200))
            img = tan_triggs_preprocessing(img)
            
            images.append(img)
            labels.append(label)
            
            # Extract folder name as person's name
            name = os.path.basename(os.path.dirname(path))
            if label not in label_names:
                label_names[label] = name
                
    return images, labels, label_names

def print_training_accuracy(model, images, labels, label_names):
    """Print per-class and overall training accuracy."""
    correct = {}
    total = {}
    
    for img, lbl in zip(images, labels):
        pred = model.predict(img)[0]
        total[lbl] = total.get(lbl, 0) + 1
        if pred == lbl:
            correct[lbl] = correct.get(lbl, 0) + 1
            
    total_all = 0
    correct_all = 0
    print("\n--- Training Set Accuracy ---")
    for lbl in sorted(total.keys()):
        name = label_names.get(lbl, "?")
        c = correct.get(lbl, 0)
        t = total[lbl]
        print(f"  {name:12s} | {c} / {t} correct ({100.0*c/t:.0f}%)")
        correct_all += c
        total_all += t
    print(f"  TOTAL        | {correct_all} / {total_all} correct ({100.0*correct_all/total_all:.0f}%)")
    print("  NOTE: Low training accuracy = bad data quality or too few images per person.")
    print("----------------------------\n")

def compute_threshold(all_images, all_labels, label_names, multiplier=1.5):
    """Compute a recognition threshold using an 80/20 per-person train/val split."""
    label_idx = {}
    for i, lbl in enumerate(all_labels):
        if lbl not in label_idx: label_idx[lbl] = []
        label_idx[lbl].append(i)
        
    train_imgs, train_lbls = [], []
    val_imgs, val_lbls = [], []
    
    for lbl, idxs in label_idx.items():
        n_val = max(1, int(len(idxs) * 0.2))
        n_train = len(idxs) - n_val
        for i in range(n_train):
            train_imgs.append(all_images[idxs[i]])
            train_lbls.append(all_labels[idxs[i]])
        for i in range(n_train, len(idxs)):
            val_imgs.append(all_images[idxs[i]])
            val_lbls.append(all_labels[idxs[i]])
            
    if len(set(train_lbls)) < 2:
        print("Not enough data for threshold calibration — using default 3000.0")
        return 3000.0
        
    tmp_model = cv2.face.FisherFaceRecognizer_create()
    tmp_model.train(train_imgs, np.array(train_lbls))
    
    label_distances = {}
    for img, lbl in zip(val_imgs, val_lbls):
        _, dist = tmp_model.predict(img)
        if lbl not in label_distances: label_distances[lbl] = []
        label_distances[lbl].append(dist)
        
    print("\n--- Per-Person Distance Statistics (held-out 20%, lower = closer) ---")
    per_label_max = []
    for lbl in sorted(label_distances.keys()):
        dists = label_distances[lbl]
        min_d, max_d = min(dists), max(dists)
        mean_d = sum(dists) / len(dists)
        std_d = np.std(dists) if len(dists) > 1 else 0.0
        name = label_names.get(lbl, "?")
        print(f"  {name:12s} | min: {min_d:7.1f}  mean: {mean_d:7.1f}  max: {max_d:7.1f}  stddev: {std_d:6.1f}  (n={len(dists)})")
        per_label_max.append(max_d)
        
    avg_max_dist = sum(per_label_max) / len(per_label_max)
    threshold = avg_max_dist * multiplier
    print(f"\nAuto threshold = {threshold:.1f}  (avg worst-case dist {avg_max_dist:.1f} x {multiplier:.1f}x margin)")
    print("----------------------------------------------------------------------\n")
    return threshold

def align_face(gray, eye_left, eye_right, out_size=200):
    """Align a face crop using eye positions to match the CropFace training preprocessing."""
    offset_w = 0.3 * out_size
    offset_h = 0.3 * out_size
    desired_dist = out_size - 2.0 * offset_w
    
    dx = eye_right[0] - eye_left[0]
    dy = eye_right[1] - eye_left[1]
    actual_dist = math.sqrt(dx*dx + dy*dy)
    if actual_dist < 1.0: return None
    
    scale = desired_dist / actual_dist
    angle = math.atan2(dy, dx) * 180.0 / math.pi
    
    eye_center = ((eye_left[0] + eye_right[0]) * 0.5,
                  (eye_left[1] + eye_right[1]) * 0.5)
                  
    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    M[0, 2] += out_size * 0.5 - eye_center[0]
    M[1, 2] += offset_h - eye_center[1]
    
    aligned = cv2.warpAffine(gray, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return aligned

def run_live_recognition(model, label_names, face_height=200, history_length=10, confidence_threshold=3000.0):
    """Live face recognition using FisherFace with eye-landmark alignment."""
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("Error loading Haar cascade detector.")
        return
        
    fm = cv2.face.createFacemarkLBF()
    facemark_loaded = False
    try:
        fm.loadModel("lbfmodel.yaml")
        facemark_loaded = True
        print("Facemark loaded: using eye-aligned crops.")
    except Exception as e:
        print(f"Warning: Landmark detector failed to load ({e}) — alignment disabled.")
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return
        
    print("Press ESC to exit live recognition.")
    face_histories = {}
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Viola-Jones detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, 0, (30, 30))
        
        # Landmark detection
        landmarks = []
        if facemark_loaded and len(faces) > 0:
            ok, detected_landmarks = fm.fit(gray, faces)
            if ok: landmarks = detected_landmarks
            
        seen_faces = {}
        for i, face in enumerate(faces):
            (x, y, w, h) = face
            face_img = None
            aligned = False
            
            if len(landmarks) > i and len(landmarks[i][0]) >= 68:
                eye_left = landmarks[i][0][36]
                eye_right = landmarks[i][0][45]
                face_img = align_face(gray, eye_left, eye_right, face_height)
                if face_img is not None: aligned = True
                
            if face_img is None:
                face_img = cv2.resize(gray[y:y+h, x:x+w], (face_height, face_height))
                
            face_img = tan_triggs_preprocessing(face_img)
            
            predicted_label, confidence = model.predict(face_img)
            
            # Simple spatial hash for tracking
            face_id = x + y * 1000
            seen_faces[face_id] = True
            
            if face_id not in face_histories:
                face_histories[face_id] = deque(maxlen=history_length)
            history = face_histories[face_id]
            history.append(predicted_label)
            
            # Majority vote
            counts = {}
            for l in history: counts[l] = counts.get(l, 0) + 1
            stable_label = max(counts, key=counts.get)
            
            predicted_name = label_names.get(predicted_label, "?")
            stable_name = label_names.get(stable_label, "?")
            
            is_confident = (confidence < confidence_threshold)
            person_name = stable_name if is_confident else f"? {predicted_name}"
            box_color = (0, 200, 0) if is_confident else (0, 140, 255)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2, cv2.LINE_AA)
            align_tag = "" if aligned else " [raw]"
            text = f"{person_name} (dist: {confidence:.0f}){align_tag}"
            cv2.putText(frame, text, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2, cv2.LINE_AA)
            
        # Clean up history for faces that left the frame
        face_histories = {fid: hist for fid, hist in face_histories.items() if fid in seen_faces}
        
        cv2.imshow("Live FisherFace Recognition (Python)", frame)
        if cv2.waitKey(30) == 27: break
        
    cap.release()
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) < 2:
        print(f"usage: python {sys.argv[0]} <csv.ext> [output_folder]")
        return
        
    fn_csv = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) == 3 else "."
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    
    images, labels, label_names = build_label_name_map(fn_csv)
    
    if len(images) <= 1:
        print("This demo needs at least 2 images to work. Please add more images to your data set!")
        return
        
    # Keep last sample for verification
    test_sample = images.pop()
    test_label = labels.pop()
    
    model = cv2.face.FisherFaceRecognizer_create()
    model.train(images, np.array(labels))
    
    print_training_accuracy(model, images, labels, label_names)
    
    # Simple prediction
    predicted_label = model.predict(test_sample)[0]
    print(f"Predicted class = {predicted_label} / Actual class = {test_label}.")
    
    # Visualization logic (Eigenfaces/Fisherfaces)
    eigenvalues = model.getEigenValues().flatten()
    eigenvectors = model.getEigenVectors()
    mean = model.getMean()
    
    height = images[0].shape[0]
    
    cv2.imwrite(os.path.join(output_folder, "mean.png"), norm_0_255(mean.reshape(height, -1)))
    
    for i in range(min(16, len(eigenvalues))):
        val = eigenvalues[i]
        ev = eigenvectors[:, i].reshape(height, -1)
        grayscale = norm_0_255(ev)
        cgrayscale = cv2.applyColorMap(grayscale, cv2.COLORMAP_BONE)
        cv2.imwrite(os.path.join(output_folder, f"fisherface_{i}.png"), cgrayscale)
        print(f"Eigenvalue #{i} = {val:.5f}")

    # Auto-threshold computation
    auto_threshold = compute_threshold(images, labels, label_names, 1.5)
    
    # Live recognition
    run_live_recognition(model, label_names, height, 10, auto_threshold)

if __name__ == "__main__":
    main()
