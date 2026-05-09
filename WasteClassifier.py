import cv2
import numpy as np
import os
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class FeatureExtractor:
    def extract(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h_mean = hsv[:, :, 0].mean()
        s_mean = hsv[:, :, 1].mean()
        v_mean = hsv[:, :, 2].mean()

        lap      = cv2.Laplacian(gray, cv2.CV_64F)
        tex_var  = lap.var()
        tex_mean = abs(lap.mean())
        tex_std  = lap.std()

        edges        = cv2.Canny(gray, 80, 150)
        edge_density = edges.mean()
        edge_std     = edges.std()

        return np.array([
            h_mean, s_mean, v_mean,
            tex_var, tex_mean, tex_std,
            edge_density, edge_std
        ])

class SimpleWasteClassifier:
    def __init__(self):
        self.features   = []
        self.labels     = []
        self.extractor  = FeatureExtractor()
        self.mean       = None
        self.std        = None
        self.cache_file = "waste_cache.pkl"

    def _normalize(self, features_array):
        return (features_array - self.mean) / (self.std + 1e-8)

    def _count_images(self, dataset_path):
        VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        all_paths = {}
        for label in os.listdir(dataset_path):
            class_folder = os.path.join(dataset_path, label)
            if not os.path.isdir(class_folder):
                continue
            paths = []
            for f in os.listdir(class_folder):
                if os.path.splitext(f)[1].lower() in VALID_EXT:
                    paths.append(os.path.join(class_folder, f))
            all_paths[label] = paths
        return all_paths

    def fit_dataset(self, dataset_path):
        current_paths = self._count_images(dataset_path)
        current_total = sum(len(v) for v in current_paths.values())

        cached_features = []
        cached_labels   = []
        cached_paths    = set()
        cached_raw      = []
        data            = {}

        if os.path.exists(self.cache_file):
            print("Cache found! Checking for new images...")
            with open(self.cache_file, "rb") as f:
                data = pickle.load(f)

            cached_features = data["features"]
            cached_labels   = data["labels"]
            cached_paths    = set(data["cached_paths"])
            cached_raw      = data.get("raw_features", [])
            cached_total    = len(cached_features)

            if cached_total == current_total:
                self.features = cached_features
                self.labels   = cached_labels
                self.mean     = data["mean"]
                self.std      = data["std"]

                class_counts = {}
                for lbl in self.labels:
                    class_counts[lbl] = class_counts.get(lbl, 0) + 1

                print(f"No new images found.")
                print(f"Loaded {len(self.features)} samples instantly!")
                print(f"Total samples: {len(self.features)}")
                for cls, count in class_counts.items():
                    print(f"  {cls}: {count}")
                return

            new_count = current_total - cached_total
            print(f"Found {new_count} new image(s)! Extracting only new ones...")
        else:
            print("No cache found. Extracting all features (first time only)...")

        new_features = []
        new_labels   = []
        new_paths    = []

        for label, paths in current_paths.items():
            for path in paths:
                if path in cached_paths:
                    continue
                img = cv2.imread(path)
                if img is None:
                    continue
                feat = self.extractor.extract(img)
                new_features.append(feat)
                new_labels.append(label)
                new_paths.append(path)

        print(f"Extracted {len(new_features)} new samples.")

        all_raw    = list(cached_raw) + new_features
        all_labels = list(cached_labels) + new_labels
        all_paths  = list(cached_paths) + new_paths

        arr           = np.array(all_raw)
        self.mean     = arr.mean(axis=0)
        self.std      = arr.std(axis=0)
        normalized    = list(self._normalize(arr))

        self.features = normalized
        self.labels   = all_labels

        with open(self.cache_file, "wb") as f:
            pickle.dump({
                "features":     normalized,
                "raw_features": all_raw,
                "labels":       all_labels,
                "mean":         self.mean,
                "std":          self.std,
                "cached_paths": all_paths,
            }, f)

        print("Cache updated!")

        class_counts = {}
        for lbl in self.labels:
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

        print(f"Total samples: {len(self.features)}")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")

    def predict(self, img, k=10):
        feat = self.extractor.extract(img)
        feat = self._normalize(feat)

        distances = [np.linalg.norm(feat - f) for f in self.features]
        k_indices = np.argsort(distances)[:k]

        vote_scores = {}
        for idx in k_indices:
            lbl   = self.labels[idx]
            dist  = distances[idx]
            score = 1 / (dist + 1e-5)
            vote_scores[lbl] = vote_scores.get(lbl, 0) + score

        best_label  = max(vote_scores, key=vote_scores.get)
        total_score = sum(vote_scores.values())
        confidence  = round((vote_scores[best_label] / total_score) * 100, 1)

        return best_label, confidence


DISPOSAL_TIPS = {
    "organic": "Put in the green compost bin",
    "plastic": "Rinse and place in blue recycle bin",
    "person":  "No disposal needed — that's a person!",
}

CLASS_COLORS_HEX = {
    "organic": "#27ae60",
    "plastic": "#3498db",
    "person":  "#e74c3c",
}

CLASS_COLORS_BGR = {
    "organic": (0, 200, 0),    # #27ae60
    "plastic": (200, 0, 0),   # #3498db
    "person":  (0, 0, 200),    # #e74c3c
}

class WasteClassifierApp:
    def __init__(self, root, classifier):
        self.root        = root
        self.classifier  = classifier
        self.cap         = None
        self.running     = False
        self.history     = []

        # Background subtraction state
        self.bg_frames      = []
        self.background     = None
        self.bg_ready       = False
        self.bg_frame_count = 30

        # Smooth bounding box state
        self.smooth_x = 0
        self.smooth_y = 0
        self.smooth_w = 0
        self.smooth_h = 0
        self.smoothing = 0.6

        # Classify cooldown
        self.frame_count    = 0
        self.classify_every = 10
        self.last_label     = ""
        self.last_conf      = 0

        self.root.title("Waste Classifier System")
        self.root.configure(bg="#1a2332")
        self.root.geometry("980x640")
        self.root.resizable(False, False)

        self._build_ui()

    def _build_ui(self):

        # ========== TOP BAR ==========
        top = tk.Frame(self.root, bg="#111c26", height=52)
        top.pack(fill="x")
        top.pack_propagate(False)

        tk.Label(top, text="Waste Classifier System",
                 font=("Arial", 20, "bold"),
                 bg="#111c26", fg="#ecf0f1").pack(side="left", padx=20, pady=14)

        self.live_badge = tk.Label(top, text="  OFFLINE  ",
                                   font=("Arial", 8, "bold"),
                                   bg="#444", fg="white",
                                   padx=8, pady=3)
        self.live_badge.pack(side="right", padx=20, pady=14)

        # ========== BODY ==========
        body = tk.Frame(self.root, bg="#1a2332")
        body.pack(fill="both", expand=True, padx=14, pady=10)

        # ---- LEFT: camera feed ----
        cam_frame = tk.Frame(body, bg="#111c26",
                             width=640, height=480)
        cam_frame.pack(side="left")
        cam_frame.pack_propagate(False)

        self.cam_label = tk.Label(cam_frame,
                                  text="Camera feed will appear here",
                                  font=("Arial", 20, "bold"),
                                  bg="#111c26", fg="#ffffff")
        self.cam_label.pack(expand=True, fill="both")

        # ---- RIGHT: info panel ----
        panel = tk.Frame(body, bg="#1a2332", width=300)
        panel.pack(side="right", fill="y", padx=(12, 0))
        panel.pack_propagate(False)

        # -- Result card --
        self._make_section_label(panel, "DETECTION RESULT")

        result_card = tk.Frame(panel, bg="#111c26")
        result_card.pack(fill="x", pady=(4, 12))

        self._make_card_row(result_card, "CLASS")
        self.result_label = tk.Label(result_card, text="—",
                                     font=("Arial", 28, "bold"),
                                     bg="#111c26", fg="#ecf0f1")
        self.result_label.pack(anchor="w", padx=16, pady=(0, 6))

        tk.Frame(result_card, bg="#1a2332", height=1).pack(fill="x")

        self._make_card_row(result_card, "CONFIDENCE")
        self.conf_label = tk.Label(result_card, text="—",
                                   font=("Arial", 22, "bold"),
                                   bg="#111c26", fg="#ecf0f1")
        self.conf_label.pack(anchor="w", padx=16, pady=(0, 6))

        tk.Frame(result_card, bg="#1a2332", height=1).pack(fill="x")

        self._make_card_row(result_card, "DISPOSAL TIP")
        self.tip_label = tk.Label(result_card, text="—",
                                  font=("Arial", 10),
                                  bg="#111c26", fg="#8899aa",
                                  wraplength=260, justify="left")
        self.tip_label.pack(anchor="w", padx=16, pady=(0, 14))

        # -- History card --
        self._make_section_label(panel, "RECENT SCANS")

        self.history_card = tk.Frame(panel, bg="#111c26")
        self.history_card.pack(fill="x", pady=(4, 12))

        self.history_inner = tk.Frame(self.history_card, bg="#111c26")
        self.history_inner.pack(fill="x", padx=16, pady=10)
        self._refresh_history()

        # ========== BOTTOM BUTTONS ==========
        btn_bar = tk.Frame(self.root, bg="#111c26", height=56)
        btn_bar.pack(fill="x", side="bottom")
        btn_bar.pack_propagate(False)

        buttons = [
            ("Start Webcam", "#27ae60", self._start_webcam),
            ("Stop Webcam",  "#e67e22", self._stop_webcam),
            ("Upload Image", "#2980b9", self._upload_image),
            ("Exit",         "#c0392b", self.root.destroy),
        ]

        for text, color, cmd in buttons:
            tk.Button(btn_bar,
                      text=text,
                      font=("Arial", 17, "bold"),
                      bg=color, fg="white",
                      activebackground=color,
                      activeforeground="white",
                      relief="flat", bd=0,
                      padx=10, pady=10,
                      cursor="hand2",
                      command=cmd
                      ).pack(side="left",
                             expand=True,
                             fill="both",
                             padx=1)

    def _make_section_label(self, parent, text):
        tk.Label(parent, text=text,
                 font=("Arial", 16, "bold"),
                 bg="#1a2332", fg="#ffffff").pack(anchor="w", pady=(6, 0))

    def _make_card_row(self, parent, text):
        tk.Label(parent, text=text,
                 font=("Arial", 14),
                 bg="#111c26", fg="#ffffff").pack(anchor="w", padx=16, pady=(10, 2))


    def _update_result(self, label, confidence):
        color = CLASS_COLORS_HEX.get(label.lower(), "#ecf0f1")
        tip   = DISPOSAL_TIPS.get(label.lower(), "")

        self.result_label.config(text=label.upper(), fg=color)
        self.conf_label.config(text=f"{confidence}%", fg=color)
        self.tip_label.config(text=tip)

        self.history.insert(0, (label, confidence))
        self.history = self.history[:6]
        self._refresh_history()

    def _refresh_history(self):
        for w in self.history_inner.winfo_children():
            w.destroy()

        if not self.history:
            tk.Label(self.history_inner,
                     text="No scans yet",
                     font=("Arial", 10),
                     bg="#111c26", fg="#2d4155").pack(anchor="w")
            return

        for i, (lbl, conf) in enumerate(self.history):
            color = CLASS_COLORS_HEX.get(lbl.lower(), "#ecf0f1")

            row = tk.Frame(self.history_inner, bg="#111c26")
            row.pack(fill="x", pady=2)

            # Colored dot indicator
            dot = tk.Label(row, text="●", font=("Arial", 8),
                           bg="#111c26", fg=color)
            dot.pack(side="left", padx=(0, 6))

            tk.Label(row, text=lbl.capitalize(),
                     font=("Arial", 10),
                     bg="#111c26", fg=color).pack(side="left")

            tk.Label(row, text=f"{conf}%",
                     font=("Arial", 10),
                     bg="#111c26", fg="#4a6278").pack(side="right")

            # Separator between rows
            if i < len(self.history) - 1:
                tk.Frame(self.history_inner,
                         bg="#1a2332", height=1).pack(fill="x", pady=1)


    def _start_webcam(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam")
            return

        # Reset background state
        self.bg_frames  = []
        self.background = None
        self.bg_ready   = False
        self.smooth_x   = 0
        self.smooth_y   = 0
        self.smooth_w   = 0
        self.smooth_h   = 0
        self.last_label = ""
        self.last_conf  = 0
        self.frame_count = 0

        self.running = True
        self.live_badge.config(text="  LIVE  ", bg="#c0392b")
        print("Learning background... keep camera still.")
        self._webcam_loop()

    def _webcam_loop(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._stop_webcam()
            return

        self.frame_count += 1
        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray    = cv2.GaussianBlur(gray, (21, 21), 0)

        # ---- Phase 1: Learn background ----
        if not self.bg_ready:
            self.bg_frames.append(gray.astype("float"))
            remaining = self.bg_frame_count - len(self.bg_frames)

            cv2.putText(display,
                        f"Learning background... keep still ({remaining})",
                        (16, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 220, 220), 2)

            if len(self.bg_frames) >= self.bg_frame_count:
                self.background = np.mean(
                    self.bg_frames, axis=0).astype("uint8")
                self.bg_ready = True
                print("Background ready!")

        # ---- Phase 2: Detect object ----
        else:
            diff      = cv2.absdiff(self.background, gray)
            _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

            # remove noise
            thresh = cv2.medianBlur(thresh, 5)

            # strengthen object area
            thresh = cv2.dilate(thresh, None, iterations=3)

            contours, _ = cv2.findContours(thresh,
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            object_found = False

            if contours:
                largest = max(contours, key=cv2.contourArea)
                area    = cv2.contourArea(largest)

                if area > 5000:
                    x, y, w, h = cv2.boundingRect(largest)
                    pad = 20
                    x   = max(0, x - pad)
                    y   = max(0, y - pad)
                    w   = min(frame.shape[1] - x, w + pad * 2)
                    h   = min(frame.shape[0] - y, h + pad * 2)

                    if self.smooth_x == 0:
                        self.smooth_x, self.smooth_y = x, y
                        self.smooth_w, self.smooth_h = w, h
                    else:
                        s = self.smoothing
                        self.smooth_x = int(self.smooth_x * s + x * (1 - s))
                        self.smooth_y = int(self.smooth_y * s + y * (1 - s))
                        self.smooth_w = int(self.smooth_w * s + w * (1 - s))
                        self.smooth_h = int(self.smooth_h * s + h * (1 - s))

                    object_found = True

                    # Classify every N frames
                    if self.frame_count % self.classify_every == 0:
                        roi = frame[self.smooth_y:self.smooth_y + self.smooth_h,
                        self.smooth_x:self.smooth_x + self.smooth_w]

                        # --- NEW: filter too small ROI ---
                        if roi.shape[0] < 50 or roi.shape[1] >= 50:
                            # --- NEW: resize ROI for consistency ---
                            roi = cv2.resize(roi, (128, 128))

                        if roi.size > 0:
                            self.last_label, self.last_conf = \
                                self.classifier.predict(roi)
                            self._update_result(self.last_label, self.last_conf)

                    # Draw dynamic bounding box
                    if self.last_label:
                        color = CLASS_COLORS_BGR.get(
                            self.last_label.lower(), (255, 255, 255))

                        # Box
                        cv2.rectangle(display,
                                      (self.smooth_x, self.smooth_y),
                                      (self.smooth_x + self.smooth_w,
                                       self.smooth_y + self.smooth_h),
                                      color, 2)

                        # Label pill (FIXED)
                        label_text = f"{self.last_label.upper()}  {self.last_conf}%"
                        (tw, th), _ = cv2.getTextSize(
                            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)

                        # smart position (above or below box)
                        if self.smooth_y - th - 14 > 10:
                            label_y = self.smooth_y - th - 14  # above
                        else:
                            label_y = self.smooth_y + self.smooth_h + 10  # below

                        # Draw background
                        cv2.rectangle(display,
                                      (self.smooth_x, label_y),
                                      (self.smooth_x + tw + 12, label_y + th + 10),
                                      color, -1)

                        # Draw text
                        cv2.putText(display, label_text,
                                    (self.smooth_x + 6, label_y + th + 2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (255, 255, 255), 2)

            if not object_found:
                self.smooth_x = self.smooth_y = 0
                self.smooth_w = self.smooth_h = 0
                cv2.putText(display,
                            "No object detected",
                            (16, 36),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (60, 80, 100), 2)

        # Push frame into tkinter label
        frame_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img_pil   = Image.fromarray(frame_rgb)
        img_pil   = img_pil.resize((640, 480), Image.LANCZOS)
        img_tk    = ImageTk.PhotoImage(img_pil)

        self.cam_label.config(image=img_tk, text="")
        self.cam_label.image = img_tk

        self.root.after(30, self._webcam_loop)

    def _stop_webcam(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cam_label.config(image="",
                              text="Camera stopped",
                              fg="#2d4155")
        self.live_badge.config(text="  OFFLINE  ", bg="#444")


    def _upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Invalid image file!")
            return

        label, conf = self.classifier.predict(img)
        self._update_result(label, conf)

        # Draw result on image and show in panel
        color = CLASS_COLORS_BGR.get(label.lower(), (255, 255, 255))
        h, w  = img.shape[:2]

        cv2.rectangle(img, (10, 10), (w - 10, h - 10), color, 3)

        label_text  = f"{label.upper()}  {conf}%"
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.rectangle(img, (10, 10), (tw + 26, th + 26), color, -1)
        cv2.putText(img, label_text,
                    (18, th + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 2)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((640, 480), Image.LANCZOS)
        img_tk  = ImageTk.PhotoImage(img_pil)

        self.cam_label.config(image=img_tk, text="")
        self.cam_label.image = img_tk


if __name__ == "__main__":
    classifier = SimpleWasteClassifier()
    classifier.fit_dataset("E:/FINALRUN/Dataset")

    root = tk.Tk()
    app  = WasteClassifierApp(root, classifier)
    root.mainloop()