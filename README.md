# Video AI Editor (CPU) — Apache-2.0

**Features**
- Optional **YOLOv8** segmentation/detection (AGPL-3.0 if installed)
- **Face detection** via OpenCV YuNet (box/blur; default = box)
- **ASR** with faster-whisper (CPU)
- **EN→AR / EN→ES** translations via Transformers (MarianMT)
- **Stacked subtitles** + **smart info cards**
- **ffmpeg** muxing to preserve audio

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements-core.txt
# Optional (YOLOv8 detection; AGPL-3.0):
# pip install -r requirements-yolo.txt
```

Ensure **ffmpeg** is on PATH. Place YuNet ONNX in `assets/` (e.g. `assets/face_detection_yunet_2023mar.onnx`).

## Config
See `features.config.json` (defaults):
```json
{
  "enable_detection": true,
  "enable_translations": true,
  "enable_infocards": true,
  "detection_style": "glow",
  "enable_face_detection": true,
  "face_style": "box",
  "face_min_conf": 0.6
}
```

## Run (example; mirrors your original command)
```bash
python src/demo_multilang_info_cpu.py   --source "samples/1.mp4"   --out    "samples/1out.mp4"   --seg_model "assets/yolov8n-seg.pt"   --imgsz 640   --seg_conf 0.50   --seg_alpha 0.55   --thickness 4   --asr_model small   --no_show
```

## Docker
```bash
docker build -t video-ai-editor:cpu .
docker run --rm -it -v %cd%:/work -w /work video-ai-editor:cpu
```

## License
- Repository code: **Apache-2.0** (see `LICENSE` and `NOTICE`).  
- Optional components (e.g., `ultralytics`): governed by their own licenses. See `THIRD_PARTY.md`.
