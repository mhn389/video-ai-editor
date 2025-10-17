# demo_multilang_info_cpu.py
# CPU-only: Segmentation glow + EN/ES/AR subtitles + smart info cards.
# Now with optional Face Detection (YuNet) controlled by config.
#
# Config file (same folder as this script): features.config.json
# {
#   "enable_detection": true,
#   "enable_translations": true,
#   "enable_infocards": true,
#   "detection_style": "glow",
#
#   "enable_face_detection": false,
#   "face_style": "box",
#   "face_min_conf": 0.6
# }
#
# Usage unchanged:
#   python demo_multilang_info_cpu.py --source "D:\\Videos\\input.mp4" --out "D:\\Videos\\output.mp4" \
#     --seg_model "yolov8n-seg.pt" --imgsz 640 --seg_conf 0.20 --seg_alpha 0.55 --thickness 4 --no_show

import os
import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO
import subprocess
import json

# ASR + Translate
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import torch  # CPU build is fine; we will force CPU use.

# Pillow + Arabic RTL support (for correct Arabic rendering)
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

# Display-only aliases (do not change OBJECT_FACTS lookup)
DISPLAY_ALIASES = {
    "mouse": "earbuds",
}

# -------------------- feature config --------------------
def load_feature_config():
    """Load toggles from features.config.json; safe defaults preserve current behavior."""
    defaults = {
        "enable_detection": True,
        "enable_translations": True,
        "enable_infocards": True,
        "detection_style": "glow",   # "glow" or "boxes"

        # Face detection (new)
        "enable_face_detection": False,   # default OFF to keep outputs unchanged
        "face_style": "box",              # "box" or "blur"
        "face_min_conf": 0.6,
    }
    cfg_path = os.path.join(os.path.dirname(__file__), "features.config.json")
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # known booleans
            for k in ("enable_detection", "enable_translations", "enable_infocards", "enable_face_detection"):
                if k in data:
                    defaults[k] = bool(data[k])
            # styles
            if "detection_style" in data:
                v = str(data["detection_style"]).lower().strip()
                if v in ("glow", "boxes"):
                    defaults["detection_style"] = v
                else:
                    print(f"[CONFIG] Unknown detection_style '{data['detection_style']}', using 'glow'.")
            if "face_style" in data:
                v = str(data["face_style"]).lower().strip()
                if v in ("box", "blur"):
                    defaults["face_style"] = v
                else:
                    print(f"[CONFIG] Unknown face_style '{data['face_style']}', using 'box'.")
            if "face_min_conf" in data:
                try:
                    defaults["face_min_conf"] = float(data["face_min_conf"])
                except Exception:
                    print("[CONFIG] face_min_conf must be a number; using 0.6")
    except Exception as e:
        print(f"[CONFIG] Failed to read features.config.json: {e}. Using defaults.")
    return defaults


# -------------------- video loader --------------------
def load_source(source_file):
    img_exts = {'.jpg','.jpeg','.png','.tif','.tiff','.dng','.webp','.mpo'}
    key = 1
    frame = None
    cap = None

    is_webcam = (str(source_file).strip() == "0")
    if not is_webcam:
        _, ext = os.path.splitext(source_file)
        image_type = ext.lower() in img_exts
    else:
        image_type = False

    if image_type:
        frame = cv2.imread(source_file)
        if frame is None:
            raise RuntimeError(f"Cannot read image: {source_file}")
        key = 0
        return image_type, key, frame, None

    # video/webcam
    if is_webcam:
        candidates = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        src = 0
    else:
        candidates = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
        src = source_file
        if not os.path.exists(source_file):
            raise RuntimeError(f"Video file does not exist: {source_file}")

    cap = None
    for backend in candidates:
        c = cv2.VideoCapture(src, backend)
        if c.isOpened():
            cap = c
            break
        c.release()

    if cap is None or not cap.isOpened():
        raise RuntimeError(f"Cannot open video/webcam source with available backends: {source_file}")

    return image_type, key, frame, cap


# -------------------- audio mux (keep original audio) --------------------
def mux_audio(src_video, processed_video, out_with_audio):
    """
    Mux original audio from src_video into processed_video.
    Uses '-map 1:a:0?' so it won't fail if source has no audio.
    Writes to out_with_audio (separate file), leaving processed_video untouched.
    """
    import shutil

    # Ensure distinct output name
    if os.path.abspath(out_with_audio) == os.path.abspath(processed_video):
        root, ext = os.path.splitext(processed_video)
        out_with_audio = f"{root}_withaudio{ext or '.mp4'}"

    if not os.path.exists(src_video):
        print("[MUX] Source video not found:", src_video)
        return False
    if not os.path.exists(processed_video):
        print("[MUX] Processed video not found:", processed_video)
        return False

    ffmpeg = "ffmpeg"  # must be on PATH
    cmd = [
        ffmpeg, "-y",
        "-i", processed_video,          # video from processed
        "-i", src_video,                # audio from original
        "-map", "0:v:0",
        "-map", "1:a:0?",               # first audio stream if present
        "-c:v", "copy",
        "-c:a", "copy",                 # keep AAC as-is
        "-shortest", "-movflags", "+faststart",
        out_with_audio
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[MUX] Muxed audio ->", out_with_audio)
        return True
    except subprocess.CalledProcessError as e:
        print("[MUX] ffmpeg mux failed.\nCMD:\n", " ".join(cmd),
              "\nSTDOUT:\n", e.stdout, "\nSTDERR:\n", e.stderr)
        return False


# -------------------- subtitles helpers --------------------
class SubtitleTrack:
    def __init__(self, segments):
        # segments: {start, end, text_en, text_ar, text_es}
        self.segments = segments
        self.idx = 0

    def text_at(self, t):
        while self.idx < len(self.segments) and t > self.segments[self.idx]['end'] + 0.05:
            self.idx += 1
        if self.idx < len(self.segments):
            s = self.segments[self.idx]
            if s['start'] - 0.05 <= t <= s['end'] + 0.05:
                return s.get('text_en'), s.get('text_es'), s.get('text_ar')
        # fallback search
        lo, hi = 0, len(self.segments) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            s = self.segments[mid]
            if t < s['start']: hi = mid - 1
            elif t > s['end']: lo = mid + 1
            else: return s.get('text_en'), s.get('text_es'), s.get('text_ar')
        return None, None, None


def transcribe_and_translate(source_path, asr_model_size="small"):
    """
    CPU-only:
      - Faster-Whisper on CPU with int8 compute_type
      - MarianMT EN->AR and EN->ES on CPU (use_safetensors for safe loading)
    """
    # ---- ASR (CPU, int8) ----
    asr = WhisperModel(asr_model_size, device="cpu", compute_type="int8")
    segments, _ = asr.transcribe(source_path, vad_filter=True)

    en_lines = []
    for seg in segments:
        en = (seg.text or "").strip()
        if en:
            en_lines.append({"start": seg.start, "end": seg.end, "text_en": en})

    if not en_lines:
        print("[ASR] No speech detected. Continuing without subtitles.")
        return []

    # ---- Translator EN->AR ----
    model_name_ar = "Helsinki-NLP/opus-mt-en-ar"
    tok_ar = MarianTokenizer.from_pretrained(model_name_ar)
    trans_ar = MarianMTModel.from_pretrained(model_name_ar, use_safetensors=True)

    # ---- Translator EN->ES ----
    model_name_es = "Helsinki-NLP/opus-mt-en-es"
    tok_es = MarianTokenizer.from_pretrained(model_name_es)
    trans_es = MarianMTModel.from_pretrained(model_name_es, use_safetensors=True)

    batch = 16

    # Arabic
    for i in range(0, len(en_lines), batch):
        chunk = en_lines[i:i+batch]
        texts = [c["text_en"] for c in chunk]
        inputs = tok_ar(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            gen = trans_ar.generate(**inputs, max_length=128)
        outs = tok_ar.batch_decode(gen, skip_special_tokens=True)
        for j, out in enumerate(outs):
            chunk[j]["text_ar"] = out

    # Spanish
    for i in range(0, len(en_lines), batch):
        chunk = en_lines[i:i+batch]
        texts = [c["text_en"] for c in chunk]
        inputs = tok_es(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            gen = trans_es.generate(**inputs, max_length=128)
        outs = tok_es.batch_decode(gen, skip_special_tokens=True)
        for j, out in enumerate(outs):
            chunk[j]["text_es"] = out

    for c in en_lines:
        c.setdefault("text_ar", "")
        c.setdefault("text_es", "")

    return en_lines


def draw_boxes_with_labels(img, boxes_xyxy, clses, confs, names, palette, thickness=2):
    """
    Draw classic YOLO-style boxes with class name and confidence.
    Font size scales with thickness so labels stay readable.
    """
    th = max(2, int(thickness))                       # border thickness
    fs = max(0.7, min(1.6, th * 0.35))               # font scale linked to thickness
    for (x1, y1, x2, y2), c, conf in zip(boxes_xyxy, clses, confs):
        color = tuple(int(v) for v in palette[int(c) % len(palette)].tolist())
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, th)

        # label
        #label = names.get(int(c), f"id_{int(c)}") if isinstance(names, dict) else str(int(c))
        #text  = f"{label} {float(conf):.2f}"
        label = names.get(int(c), f"id_{int(c)}") if isinstance(names, dict) else str(int(c))
        disp  = DISPLAY_ALIASES.get(label, label)   # <-- display alias
        text  = f"{disp} {float(conf):.2f}"

        (tw, th_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(2, th // 2))
        bx, by = x1, max(0, y1 - th_text - 6)
        cv2.rectangle(img, (bx, by), (bx + tw + 12, by + th_text + 6), color, -1)
        cv2.putText(img, text, (bx + 6, by + th_text),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(2, th // 2), cv2.LINE_AA)


def draw_subtitles_triple(frame, line_en, line_es, line_ar,
                          pad=18, thickness=2,
                          font_en_path=None, font_es_path=None, font_ar_path=None):
    """
    Bottom-centered stacked subtitles:
      1) English (LTR)
      2) Spanish (LTR)
      3) Arabic (RTL, shaped)
    Uses Pillow so Arabic renders correctly (cv2.putText can't shape/RTL).
    """
    if not (line_en or line_es or line_ar):
        return frame

    # Choose fonts (adjust paths if needed)
    if font_en_path is None:
        font_en_path = r"C:\Windows\Fonts\segoeui.ttf"
    if font_es_path is None:
        font_es_path = r"C:\Windows\Fonts\segoeui.ttf"
    if font_ar_path is None:
        font_ar_path = r"C:\Windows\Fonts\segoeui.ttf"  # or a dedicated Arabic font

    size_en = 26
    size_es = 26
    size_ar = 28

    try: font_en = ImageFont.truetype(font_en_path, size=size_en)
    except: font_en = ImageFont.load_default()
    try: font_es = ImageFont.truetype(font_es_path, size=size_es)
    except: font_es = ImageFont.load_default()
    try: font_ar = ImageFont.truetype(font_ar_path, size=size_ar)
    except: font_ar = ImageFont.load_default()

    # Arabic shaping + bidi
    line_ar_rtl = None
    if line_ar:
        line_ar_rtl = get_display(arabic_reshaper.reshape(line_ar))

    # Convert to Pillow
    h, w = frame.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(img_pil)
    stroke = max(1, thickness)

    def text_size(txt, font):
        if not txt: return (0, 0)
        box = draw.textbbox((0,0), txt, font=font, stroke_width=stroke)
        return (box[2]-box[0], box[3]-box[1])

    en_w, en_h = text_size(line_en, font_en)
    es_w, es_h = text_size(line_es, font_es)
    ar_w, ar_h = text_size(line_ar_rtl, font_ar) if line_ar_rtl else (0, 0)

    gap = 6
    lines = [(line_en, font_en, en_w, en_h),
             (line_es, font_es, es_w, es_h),
             (line_ar_rtl, font_ar, ar_w, ar_h)]
    lines = [L for L in lines if L[0]]  # only those that exist

    total_h = sum(L[3] for L in lines) + pad*2 + gap*(len(lines)-1)
    max_w   = max((L[2] for L in lines), default=0)

    # Background (rounded) box
    box_w = max_w + 24
    box_h = total_h
    x = max(10, (w - box_w) // 2)
    y = h - box_h - 10

    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    try:
        od.rounded_rectangle([x, y, x + box_w, y + box_h], radius=10, fill=(0,0,0,180))
    except:
        od.rectangle([x, y, x + box_w, y + box_h], fill=(0,0,0,180))
    img_pil = Image.alpha_composite(img_pil, overlay)
    draw = ImageDraw.Draw(img_pil)

    # Draw stacked lines centered
    cur_y = y + pad
    for txt, font, tw, th in lines:
        tx = x + (box_w - tw)//2
        draw.text((tx, cur_y), txt, font=font,
                  fill=(255,255,255), stroke_width=stroke, stroke_fill=(0,0,0))
        cur_y += th + gap

    # Back to OpenCV
    return cv2.cvtColor(np.asarray(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)


# -------------------- info cards --------------------
OBJECT_FACTS = {
    "cell phone": "Best guess: iPhone 13, 6.1 Super Retina XDR OLED; Face ID; 5G; MagSafe; dual 12 MP. Could be one of: iPhone 12 / 13 / 14 (non-Pro)",
    "laptop":     "MacBook Pro 13” (Touch Bar) 13.3 Retina 2560x1600; Touch ID; Thunderbolt (USB C) ports, 2016 to 2020 generation, including late-2020 M1",
    "microphone": "Mic — cardioid; ~1 palm distance",
    #"cup":        "Cup — ~250 ml; watch reflections",
    "bottle":     "Bottle — glare may raise exposure",
    "keyboard":   "Keyboard — LED highlights may flicker",
    "book":       "Book — good focus target",
}

def _wrap_text_lines(text, font_face, font_scale, thickness, max_width_px):
    """
    Simple greedy word-wrap using cv2.getTextSize.
    Returns a list of lines that each fit within max_width_px.
    """
    words = (text or "").split()
    if not words:
        return []

    lines = []
    cur = words[0]
    for w in words[1:]:
        trial = cur + " " + w
        (tw, th), _ = cv2.getTextSize(trial, font_face, font_scale, thickness)
        if tw <= max_width_px:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _rects_intersect(a, b, margin=6):
    """Return True if rectangles a and b intersect. Rect = (x, y, w, h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    # add a small margin so boxes don't visually touch
    return not (ax2 + margin <= bx or bx2 + margin <= ax or ay2 + margin <= by or by2 + margin <= ay)


def _clamp_card_rect(x, y, cw, ch, fw, fh, top_limit=0, bottom_reserved=0):
    """Clamp a card (x,y,cw,ch) inside frame (fw,fh), keeping a bottom reserved band."""
    x = max(5, min(x, fw - cw - 5))
    max_y = fh - bottom_reserved - ch - 5
    y = max(top_limit + 5, min(y, max_y))
    return x, y


def _try_place_card(pref_positions, cw, ch, fw, fh, placed, top_limit=0, bottom_reserved=0):
    """Try candidate positions; return first non-overlapping clamped rect or None."""
    for px, py in pref_positions:
        cx, cy = _clamp_card_rect(px, py, cw, ch, fw, fh, top_limit, bottom_reserved)
        cand = (cx, cy, cw, ch)
        if all(not _rects_intersect(cand, r) for r in placed):
            return cand
    return None


def _slide_to_clear(start_rect, fw, fh, placed, direction="down",
                    step=12, max_steps=120, top_limit=0, bottom_reserved=0):
    """From a starting rect, slide until it clears collisions or give up."""
    cx, cy, cw, ch = start_rect
    for _ in range(max_steps):
        if direction == "down":
            cy += step
        elif direction == "up":
            cy -= step
        cx, cy = _clamp_card_rect(cx, cy, cw, ch, fw, fh, top_limit, bottom_reserved)
        cand = (cx, cy, cw, ch)
        if all(not _rects_intersect(cand, r) for r in placed):
            return cand
    return None


def draw_info_cards(image, boxes_xyxy, clses, names, max_cards=2):
    """
    Dynamic info cards with collision avoidance:
      - word-wrapped text
      - try positions: above → below → right → left
      - clamp inside frame and avoid subtitle band
      - slide to clear if all collide; skip if impossible
    """
    h, w = image.shape[:2]
    count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Same visual style as before
    font_scale = 0.55
    text_th = 2
    pad_x, pad_y = 8, 8
    line_gap = 6
    max_width_px = int(0.55 * w)        # wrap threshold ~55% of frame width
    bg_color = (40, 40, 40)
    fg_color = (255, 255, 255)

    # Reserve a bottom band to avoid covering subtitles (adjust if needed)
    reserved_bottom = int(0.14 * h)     # ~14% of frame height
    placed_rects = [(0, h - reserved_bottom, w, reserved_bottom)]  # treat as a blocking rect

    for (x1, y1, x2, y2), c in zip(boxes_xyxy, clses):
        if count >= max_cards:
            break

        label = names.get(int(c), f"id_{int(c)}") if isinstance(names, dict) else str(int(c))
        fact = OBJECT_FACTS.get(label, None)
        if not fact:
            continue

        text = f"{label}: {fact}"

        # Word-wrap
        lines = _wrap_text_lines(text, font, font_scale, text_th, max_width_px)
        if not lines:
            continue

        # Measure wrapped block
        line_sizes = [cv2.getTextSize(t, font, font_scale, text_th)[0] for t in lines]
        block_w = max((sz[0] for sz in line_sizes), default=0) + pad_x * 2
        block_h = sum((sz[1] for sz in line_sizes)) + pad_y * 2 + line_gap * (len(lines) - 1)

        # Preferred positions relative to object box
        # above, below, right, left (add small gaps)
        pref = [
            (int(x1),           int(y1) - 10 - block_h),  # above
            (int(x1),           int(y2) + 10),            # below
            (int(x2) + 10,      int(y1)),                 # right
            (int(x1) - 10 - block_w, int(y1)),            # left
        ]

        # Try direct placements
        spot = _try_place_card(pref, block_w, block_h, w, h, placed_rects,
                               top_limit=0, bottom_reserved=reserved_bottom)

        # If all collide, try sliding from "below" first, then from "above"
        if spot is None:
            # start below
            start_below = _clamp_card_rect(pref[1][0], pref[1][1], block_w, block_h, w, h,
                                           0, reserved_bottom)
            cand = (start_below[0], start_below[1], block_w, block_h)
            slid = _slide_to_clear(cand, w, h, placed_rects, direction="down",
                                   step=12, max_steps=120, top_limit=0, bottom_reserved=reserved_bottom)
            if slid is None:
                # try from above, sliding up
                start_above = _clamp_card_rect(pref[0][0], pref[0][1], block_w, block_h, w, h,
                                               0, reserved_bottom)
                cand = (start_above[0], start_above[1], block_w, block_h)
                slid = _slide_to_clear(cand, w, h, placed_rects, direction="up",
                                       step=12, max_steps=120, top_limit=0, bottom_reserved=reserved_bottom)
            spot = slid

        # If still no space, skip this card gracefully
        if spot is None:
            continue

        cx, cy, cw, ch = spot
        placed_rects.append(spot)

        # Draw background
        cv2.rectangle(image, (cx, cy), (cx + cw, cy + ch), bg_color, -1)

        # Draw wrapped lines
        cur_y = cy + pad_y
        for (tline, (tw, th_line)) in zip(lines, line_sizes):
            cv2.putText(image, tline, (cx + pad_x, cur_y + th_line),
                        font, font_scale, fg_color, text_th, cv2.LINE_AA)
            cur_y += th_line + line_gap

        count += 1


# -------------------- Face Detection (YuNet) helpers --------------------
def _get_yunet_path():
    """Default YuNet ONNX path (place file next to this script)."""
    return os.path.join(os.path.dirname(__file__), "face_detection_yunet_2023mar.onnx")

def _init_yunet_detector(input_w, input_h, score_thr=0.6):
    """Create a YuNet detector if available; otherwise return None."""
    model_path = _get_yunet_path()
    if not hasattr(cv2, "FaceDetectorYN_create"):
        print("[FACE] OpenCV was built without FaceDetectorYN. Face detection disabled.")
        return None
    if not os.path.exists(model_path):
        print(f"[FACE] YuNet ONNX not found at: {model_path}\n"
              f"       Download from OpenCV Zoo and place it there. Continuing without faces.")
        return None
    try:
        det = cv2.FaceDetectorYN_create(
            model=model_path,
            config="",                     # no *.yaml needed
            input_size=(int(input_w), int(input_h)),
            score_threshold=float(score_thr),
            nms_threshold=0.3,
            top_k=5000
        )
        return det
    except Exception as e:
        print("[FACE] Failed to initialize YuNet:", e)
        return None

def _yunet_detect(det, img, face_style="box", color=(0, 255, 255)):
    """
    Run YuNet on BGR image. Draw box+label OR blur faces in-place.
    Returns the number of faces.
    """
    if det is None:
        return 0
    H, W = img.shape[:2]
    try:
        det.setInputSize((W, H))
    except Exception:
        pass

    faces = None
    try:
        ret = det.detect(img)
        # API returns either (retval, faces) or faces
        if isinstance(ret, tuple):
            faces = ret[1]
        else:
            faces = ret
    except Exception as e:
        # If inference fails, skip
        # print("[FACE] detect error:", e)
        return 0

    if faces is None or len(faces) == 0:
        return 0

    # YuNet output: Nx15 -> x,y,w,h, 5 landmarks (x,y)*5, score
    count = 0
    for f in faces:
        x, y, w, h = f[0:4].astype(np.int32)
        score = float(f[-1]) if len(f) >= 15 else 0.0

        # clip box to frame
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W - 1, x + w), min(H - 1, y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        if face_style == "blur":
            roi = img[y1:y2, x1:x2]
            if roi.size > 0:
                # kernel proportional to box size, odd integers
                kx = max(15, (w // 7) | 1)
                ky = max(15, (h // 7) | 1)
                roi_blur = cv2.GaussianBlur(roi, (kx, ky), 0)
                img[y1:y2, x1:x2] = roi_blur
        else:
            # draw box + label
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"face {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            bx, by = x1, max(0, y1 - th - 6)
            cv2.rectangle(img, (bx, by), (bx + tw + 10, by + th + 6), color, -1)
            cv2.putText(img, label, (bx + 5, by + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        count += 1

    return count


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, required=True, help="Video/Image path or '0' for webcam")
    ap.add_argument("--out", type=str, required=True, help="Output .mp4 path (must differ from source if a file)")
    ap.add_argument("--seg_model", type=str, default="yolov8n-seg.pt", help="Ultralytics segmentation model")
    ap.add_argument("--seg_conf", type=float, default=0.20, help="Segmentation confidence")
    ap.add_argument("--seg_alpha", type=float, default=0.55, help="Glow strength [0..1]")
    ap.add_argument("--imgsz", type=int, default=640, help="Segmentation inference size")
    ap.add_argument("--thickness", type=int, default=4, help="Contour thickness")
    ap.add_argument("--no_show", action="store_true", help="Do not display window")
    ap.add_argument("--asr_model", type=str, default="small", help="Faster-Whisper size: tiny/base/small/medium/large-v3")
    args = ap.parse_args()

    # Load feature toggles
    cfg = load_feature_config()
    FEAT_DET   = cfg["enable_detection"]
    FEAT_SUBS  = cfg["enable_translations"]
    FEAT_INFO  = cfg["enable_infocards"]
    DET_STYLE  = cfg["detection_style"]  # "glow" or "boxes"

    FEAT_FACE  = cfg["enable_face_detection"]
    FACE_STYLE = cfg["face_style"]       # "box" or "blur"
    FACE_CONF  = float(cfg["face_min_conf"])

    print(f"[FEATURE] detection_style = {DET_STYLE}")
    if FEAT_FACE:
        print(f"[FEATURE] face_detection = ON  ({FACE_STYLE}, min_conf={FACE_CONF})")
    else:
        print(f"[FEATURE] face_detection = OFF")

    # Prevent overwriting the source
    if os.path.abspath(args.out) == os.path.abspath(args.source):
        root, ext = os.path.splitext(args.out)
        args.out = f"{root}_OUT{ext or '.mp4'}"

    # ---- YOLO model (CPU) ----
    seg_model = None
    device = "cpu"
    if FEAT_DET:
        seg_model = YOLO(args.seg_model)
        seg_model.to(device)
        print("Device:", device)
    else:
        print("[FEATURE] Detection disabled via config.")

    # Load video
    image_type, key, frame, cap = load_source(args.source)

    # Probe first frame/fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    if not image_type:
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read from source: {args.source}")
        frame = first_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        got_fps = cap.get(cv2.CAP_PROP_FPS)
        if got_fps and got_fps > 0:
            fps = got_fps

    # Prepare writer
    if frame is None:
        raise RuntimeError("Could not initialize frame size for writer.")
    h, w = frame.shape[:2]
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    # Face detector init (after we know size)
    face_detector = None
    if FEAT_FACE:
        face_detector = _init_yunet_detector(w, h, score_thr=FACE_CONF)

    # -------- Transcribe + Translate (if enabled) --------
    if FEAT_SUBS:
        print("Transcribing + translating (CPU)…")
        subs = transcribe_and_translate(args.source, asr_model_size=args.asr_model)
        track = SubtitleTrack(subs) if subs else SubtitleTrack([])
    else:
        print("[FEATURE] Translations disabled via config.")
        track = SubtitleTrack([])

    # Colors for classes
    rng = np.random.default_rng(2024)
    palette = rng.integers(0, 255, size=(1000, 3), dtype=np.uint8)

    frame_idx = 0
    while True:
        if image_type:
            img = frame.copy()
            grabbed = img is not None
            t_sec = 0.0
        else:
            grabbed, frame = cap.read()
            if not grabbed:
                break
            img = frame.copy()
            t_sec = frame_idx / max(fps, 1e-6)

        H, W = img.shape[:2]

        # ---- Detection drawing (glow OR boxes) ----
        if FEAT_DET and seg_model is not None:
            res = seg_model.predict(
                img, imgsz=args.imgsz, conf=args.seg_conf,
                device=device, half=False, verbose=False
            )[0]

            # Common outputs
            names = seg_model.model.names if hasattr(seg_model.model, "names") else {}
            boxes = res.boxes.xyxy.cpu().numpy().astype(int) if res.boxes is not None else []
            clses = res.boxes.cls.cpu().numpy().astype(int)  if res.boxes is not None else []
            confs = res.boxes.conf.cpu().numpy()             if res.boxes is not None else []

            if DET_STYLE == "glow":
                # Segmentation glow path (unchanged visual effect)
                if res.masks is not None and len(res.masks.data) > 0:
                    masks = res.masks.data.cpu().numpy()  # (N, hm, wm)
                    overlay = img.copy()

                    # We'll collect label anchors per mask index
                    label_items = []  # list of (x_anchor, y_anchor, class_id, conf)

                    for i, m in enumerate(masks):
                        m = (m > 0.5).astype(np.uint8)
                        if m.shape[0] != H or m.shape[1] != W:
                            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

                        # color from class id if available, else default 0
                        c = int(clses[i]) if i < len(clses) else 0
                        color = palette[c % len(palette)]
                        sel = m.astype(bool)
                        overlay[sel] = (0.6 * overlay[sel] + 0.4 * color).astype(np.uint8)

                        # contours (for glow outline + label anchor if needed)
                        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            cv2.drawContours(overlay, cnts, -1, color.tolist(), max(2, args.thickness))
                            # choose largest contour for a stable anchor if we don't have a box
                            largest = max(cnts, key=cv2.contourArea)
                            bx, by, bw, bh = cv2.boundingRect(largest)
                        else:
                            bx = by = 0
                            bw = bh = 0

                        # determine label anchor (prefer YOLO box if aligned)
                        if i < len(boxes):
                            x1, y1, x2, y2 = boxes[i]
                            ax = int(x1)
                            ay = max(0, int(y1) - 10)
                        elif bw > 0 and bh > 0:
                            ax = int(bx)
                            ay = max(0, int(by) - 10)
                        else:
                            continue

                        conf = float(confs[i]) if i < len(confs) else 1.0
                        label_items.append((ax, ay, c, conf))

                    # blend overlay back
                    img = cv2.addWeighted(overlay, args.seg_alpha, img, 1.0 - args.seg_alpha, 0)

                    # --- NEW: draw labels after blending so they remain crisp ---
                    for ax, ay, c, conf in label_items:
                        color = palette[c % len(palette)].tolist()
                        cls_name = names.get(int(c), f"id_{int(c)}") if isinstance(names, dict) else str(int(c))
                        text = f"{cls_name} {conf:.2f}"
                        #(tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        fs = max(0.7, min(1.6, max(2, int(args.thickness)) * 0.35))
                        text_th = max(2, int(args.thickness) // 2)

                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(2, int(args.thickness) // 2))
                        by = max(0, ay - th - 6)
                        cv2.rectangle(img, (ax, by), (ax + tw + 10, by + th + 6), color, -1)
                        cv2.putText(img, text, (ax + 5, by + th),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            elif DET_STYLE == "boxes":
                # Classic YOLO boxes + labels
                if len(boxes) > 0:
                    draw_boxes_with_labels(img, boxes, clses, confs, names, palette, thickness=args.thickness)

            # Smart info cards (independent of style, but needs boxes)
            if FEAT_INFO and len(boxes) > 0:
                draw_info_cards(img, boxes, clses, names, max_cards=3)

        # ---- Face detection (YuNet): box or blur ----
        if FEAT_FACE and face_detector is not None:
            _yunet_detect(
                face_detector,
                img,
                face_style=FACE_STYLE,
                color=(0, 255, 255)  # yellow-ish for faces
            )

        # ---- Subtitles (if enabled) ----
        if FEAT_SUBS:
            en, es, ar = track.text_at(t_sec) if track else (None, None, None)
            if en or es or ar:
                img = draw_subtitles_triple(
                    img, en, es, ar,
                    pad=18, thickness=2,
                )

        # Write / show
        writer.write(img)
        if not args.no_show:
            cv2.imshow("Demo (CPU) — Glow + Subtitles + Info", img)
            if cv2.waitKey(1) == ord('q'):
                break

        frame_idx += 1
        if image_type:
            break

    writer.release()
    if cap: cap.release()
    cv2.destroyAllWindows()

    # Mux original audio back (separate file with suffix _withaudio.mp4)
    mux_audio(args.source, args.out, os.path.splitext(args.out)[0] + "_withaudio.mp4")

    print("\nDone:", args.out)


if __name__ == "__main__":
    main()
