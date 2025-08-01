
# YOLO Multi-Model Tracker GUI

This is a Tkinter-based graphical interface for running multiple YOLO models simultaneously on video or webcam input. It supports object detection, real-time visualization, model-specific customization, and tracking modes with options like BOTSort and ByteTrack.

---

## 🧰 Features

- 🧠 **Multi-Model Loading**: Track with multiple YOLOv8 models at once.
- 🎨 **Color + Confidence Per Model**: Customize box color and detection confidence per model.
- 🎥 **Video & Webcam Input**: Supports local video files or webcam feeds.
- ✂️ **Bounding Box Combination**: Merge overlapping boxes across models by IoU.
- 👁 **Live View Options**: Customize font size, box thickness, FPS display, and label overrides.
- 🧭 **Tracking Modes**: Predict-only or use trackers like BOTSort and ByteTrack.
- 📦 **Dual Processing**: Optionally process both original and inverted images for enhanced detection.
- ⚙️ **Live Updates**: Apply style/threshold changes without restarting the tracker.

---

## 🖥 Requirements

- Python 3.8+
- `ultralytics` (YOLOv8)
- `opencv-python`
- `numpy`

Install with:
```bash
pip install ultralytics opencv-python numpy
```

---

## 🚀 How to Use

1. Save the script as `yolo_tracker_gui.py`.
2. Run it:
```bash
python yolo_tracker_gui.py
```
3. Use the GUI to:
   - Add models with custom colors and confidence
   - Choose a video file or webcam
   - Adjust detection/tracking settings
   - Click **Run Tracker**

---

## 🧪 Tracker Modes

- `predict`: YOLO inference only
- `track`: ByteTrack (default tracker from YOLO)
- `botsort`: Advanced tracking with BOTSort

---

## 🧠 Smart Features

- **ID Override**: Replace class names per ID via a JSON-style input.
- **Color Override**: Customize ID-based box colors.
- **Cropping/Scaling**: Choose how input is resized for YOLO.
- **Real-Time Settings**: Apply new styles, FPS, or display toggles live.

---

## 🛠 Configuration

- Config is auto-saved to `tracker_config.json`
- Loaded on next launch with your previous settings.

---

## 🎨 Example Config Overrides

**Name Override:**
```json
{ "1": "Car", "2": "Person" }
```

**Color Override:**
```json
{ "1": [255, 0, 0], "2": [0, 255, 0] }
```

---

## 🧩 Notes

- Each model runs independently on GPU (`.to('cuda')`)
- Crop area shown in red when enabled
- Merged boxes appear in **magenta**

---

## 📜 License

Provided by **easyvfx**. Use freely for personal or professional purposes.

---

## 🤝 Contributions

Open to feature suggestions, bug fixes, and performance improvements!
