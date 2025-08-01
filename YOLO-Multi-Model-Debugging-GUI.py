import json
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import threading
from ultralytics import YOLO
import cv2
import numpy as np
import ast
import time

# --- Constants ---
CONFIG_PATH = 'tracker_config.json'
WINDOW_NAME = 'YOLO Multi-Model Tracker'
VIEWER_WIDTH = 1920
VIEWER_HEIGHT = 1080

# --- Global Variables for Threading ---
tracking_thread = None
stop_event = threading.Event()
live_config = {}


# --- Configuration Management ---
def load_config():
    """Loads configuration from a JSON file, providing defaults for missing keys."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            config = {}
    else:
        config = {}

    # Define default configuration
    defaults = {
        'models': [],
        'video_path': '',
        'conf_threshold': 0.25,
        'use_webcam': False,
        'webcam_id': 0,
        'dual_processing': False,
        'yolo_input_size': 640,
        'display_id': True,
        'resize_mode': 'Scale',
        'box_thickness': 2,
        'font_size': 0.6,
        'override_enabled': False,
        'override_text': '',
        'override_color_enabled': False,
        'override_color_text': '',
        'show_crop_area': False,
        'tracking_mode': 'predict',
        'iou_threshold': 0.5,
        'detections_per_second': 10,
        'show_fps': True,
        'combine_bboxes': False,
        'combine_iou_threshold': 0.6,
    }

    # Merge loaded config with defaults
    defaults.update(config)
    if 'per_model_conf' not in defaults:
        defaults['per_model_conf'] = False
    return defaults


def save_config(cfg):
    """Saves the configuration to a JSON file."""
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=4)


def select_file(var, filetypes):
    """Opens a file dialog to select a file and sets the path to a Tkinter variable."""
    path = filedialog.askopenfilename(title='Select file', filetypes=filetypes)
    if path:
        var.set(path)


# --- UI Builder ---
def build_ui():
    """Builds the main Tkinter GUI for configuration."""
    cfg = load_config()
    root = tk.Tk()
    root.title('YOLO Tracker Configuration')
    root.resizable(False, False)

    global live_config
    live_config = cfg.copy()

    # --- Tkinter Variables ---
    model_list_data = cfg.get('models', [])
    video_var = tk.StringVar(value=cfg.get('video_path', ''))
    conf_var = tk.DoubleVar(value=cfg.get('conf_threshold', 0.25))
    use_webcam_var = tk.BooleanVar(value=cfg.get('use_webcam', False))
    webcam_id_var = tk.IntVar(value=cfg.get('webcam_id', 0))
    dual_process_var = tk.BooleanVar(value=cfg.get('dual_processing', False))
    input_size_var = tk.IntVar(value=cfg.get('yolo_input_size', 640))
    display_id_var = tk.BooleanVar(value=cfg.get('display_id', True))
    resize_mode_var = tk.StringVar(value=cfg.get('resize_mode', 'Scale'))
    thickness_var = tk.IntVar(value=cfg.get('box_thickness', 2))
    font_size_var = tk.DoubleVar(value=cfg.get('font_size', 0.6))
    override_enabled_var = tk.BooleanVar(value=cfg.get('override_enabled', False))
    override_text_var = tk.StringVar(value=cfg.get('override_text', ''))
    override_color_enabled_var = tk.BooleanVar(value=cfg.get('override_color_enabled', False))
    override_color_text_var = tk.StringVar(value=cfg.get('override_color_text', ''))
    show_crop_area_var = tk.BooleanVar(value=cfg.get('show_crop_area', False))
    tracking_mode_var = tk.StringVar(value=cfg.get('tracking_mode', 'predict'))
    iou_var = tk.DoubleVar(value=cfg.get('iou_threshold', 0.5))
    detections_per_second_var = tk.IntVar(value=cfg.get('detections_per_second', 10))
    show_fps_var = tk.BooleanVar(value=cfg.get('show_fps', True))
    per_model_conf_var = tk.BooleanVar(value=cfg.get('per_model_conf', False))
    combine_bboxes_var = tk.BooleanVar(value=cfg.get('combine_bboxes', False))
    combine_iou_var = tk.DoubleVar(value=cfg.get('combine_iou_threshold', 0.6))

    new_model_path_var = tk.StringVar()
    new_model_color_var = tk.Variable(value=(0, 255, 0)) # Store as tuple, not string
    new_model_conf_var = tk.DoubleVar(value=cfg.get('conf_threshold', 0.25))

    # --- Model Management Frame ---
    model_frame = tk.LabelFrame(root, text="Model Management", padx=5, pady=5)
    model_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

    model_listbox = tk.Listbox(model_frame, height=4, width=80)
    model_listbox.grid(row=0, column=0, columnspan=3, padx=5, pady=5)

    def update_model_listbox():
        model_listbox.delete(0, tk.END)
        for model_info in model_list_data:
            path = model_info.get('path', 'N/A')
            color = model_info.get('color', 'N/A')
            conf = model_info.get('conf', 'N/A')
            model_listbox.insert(tk.END, f"Path: {os.path.basename(path)}, Color: {color}, Conf: {conf}")

    tk.Label(model_frame, text='Model Path:').grid(row=1, column=0, sticky='w', padx=5)
    model_path_entry = tk.Entry(model_frame, textvariable=new_model_path_var, width=50)
    model_path_entry.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5)
    model_browse_btn = tk.Button(model_frame, text='Browse...', command=lambda: select_file(new_model_path_var,
                                                                                            [('Model files', '*.pt'),
                                                                                             ('All files', '*.*')]))
    model_browse_btn.grid(row=2, column=2, padx=5)

    # --- NEW: Color Chooser ---
    tk.Label(model_frame, text='Box Color:').grid(row=3, column=0, sticky='w', padx=5, pady=5)
    color_swatch = tk.Label(model_frame, text="", bg="#00ff00", width=4, relief='sunken')
    color_swatch.grid(row=3, column=1, sticky='w', padx=(0, 5))

    def choose_color():
        # Ask for color, returns ((r, g, b), '#rrggbb')
        color_code = colorchooser.askcolor(title="Choose color")
        if color_code and color_code[0]:
            rgb = color_code[0]
            # Convert RGB from color picker to BGR for OpenCV
            bgr_color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            new_model_color_var.set(bgr_color)
            # Update the swatch background color
            hex_color = color_code[1]
            color_swatch.config(bg=hex_color)

    color_choose_btn = tk.Button(model_frame, text='Choose Color...', command=choose_color)
    color_choose_btn.grid(row=3, column=1, sticky='w', padx=(40, 5))
    # --- END NEW ---

    tk.Label(model_frame, text='Confidence:').grid(row=4, column=0, sticky='w', padx=5)
    model_conf_entry = tk.Entry(model_frame, textvariable=new_model_conf_var, width=10)
    model_conf_entry.grid(row=4, column=1, sticky='w', padx=5)

    def add_model():
        path = new_model_path_var.get().strip()
        color = new_model_color_var.get()
        conf = new_model_conf_var.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid model file.")
            return
        
        if not (isinstance(color, tuple) and len(color) == 3):
            messagebox.showerror("Error", "Please select a valid color.")
            return

        model_list_data.append({'path': path, 'color': color, 'conf': conf})
        update_model_listbox()
        new_model_path_var.set('')

    def update_model():
        selected_indices = model_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select a model to update.")
            return
        index = selected_indices[0]

        path = new_model_path_var.get().strip()
        color = new_model_color_var.get()
        conf = new_model_conf_var.get()

        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid model file.")
            return

        if not (isinstance(color, tuple) and len(color) == 3):
            messagebox.showerror("Error", "Please select a valid color.")
            return

        model_list_data[index] = {'path': path, 'color': color, 'conf': conf}
        update_model_listbox()

    def remove_model():
        selected_indices = model_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select a model to remove.")
            return
        for i in sorted(selected_indices, reverse=True):
            del model_list_data[i]
        update_model_listbox()

    add_model_btn = tk.Button(model_frame, text='Add Model', command=add_model)
    add_model_btn.grid(row=5, column=0, pady=5)
    remove_model_btn = tk.Button(model_frame, text='Remove Selected', command=remove_model)
    remove_model_btn.grid(row=5, column=1, pady=5)
    update_model_btn = tk.Button(model_frame, text='Update Selected', command=update_model)
    update_model_btn.grid(row=5, column=2, pady=5)

    def on_model_select(evt):
        w = evt.widget
        if not w.curselection():
            return
        index = int(w.curselection()[0])
        model_info = model_list_data[index]
        new_model_path_var.set(model_info.get('path', ''))
        
        # Update color var and swatch
        bgr_color = model_info.get('color', (0, 255, 0))
        new_model_color_var.set(bgr_color)
        hex_color = f"#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}"
        color_swatch.config(bg=hex_color)

        new_model_conf_var.set(model_info.get('conf', 0.25))

    model_listbox.bind('<<ListboxSelect>>', on_model_select)

    # --- Video Source Frame ---
    video_frame = tk.LabelFrame(root, text="Video Source", padx=5, pady=5)
    video_frame.grid(row=1, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

    use_webcam_check = tk.Checkbutton(video_frame, text="Use Webcam", variable=use_webcam_var)
    use_webcam_check.grid(row=0, column=0, sticky='w', padx=5)

    tk.Label(video_frame, text='Webcam ID:').grid(row=0, column=1, sticky='e', padx=5)
    webcam_id_entry = tk.Entry(video_frame, textvariable=webcam_id_var, width=5)
    webcam_id_entry.grid(row=0, column=2, sticky='w', padx=5)

    tk.Label(video_frame, text='Video Path:').grid(row=1, column=0, sticky='e', padx=(5, 18))
    video_path_entry = tk.Entry(video_frame, textvariable=video_var, width=50)
    video_path_entry.grid(row=1, column=1, columnspan=2, sticky='w', padx=5)
    video_browse_btn = tk.Button(video_frame, text='Browse...', command=lambda: select_file(video_var, [
        ('Video files', '*.mp4;*.mkv;*.avi'), ('All files', '*.*')]))
    video_browse_btn.grid(row=1, column=3, padx=5)

    # --- Processing Options Frame ---
    options_frame = tk.LabelFrame(root, text="Processing Options", padx=5, pady=5)
    options_frame.grid(row=2, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

    tk.Label(options_frame, text='Detections per Second:').grid(row=0, column=0, sticky='w', padx=5, pady=5)
    dps_scale = tk.Scale(options_frame, from_=1, to=60, orient=tk.HORIZONTAL, variable=detections_per_second_var,
                         length=200)
    dps_scale.grid(row=0, column=1, columnspan=2, sticky='w', padx=5)

    tk.Label(options_frame, text='Confidence Threshold:').grid(row=1, column=0, sticky='e', padx=5, pady=5)
    conf_entry = tk.Entry(options_frame, textvariable=conf_var, width=10)
    conf_entry.grid(row=1, column=1, sticky='w', padx=5)

    per_model_conf_check = tk.Checkbutton(options_frame, text="Enable Per-Model Confidence",
                                          variable=per_model_conf_var)
    per_model_conf_check.grid(row=1, column=2, sticky='w', padx=5)

    tracking_frame = tk.LabelFrame(options_frame, text="Tracking Options", padx=5, pady=5)
    tracking_frame.grid(row=2, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

    tk.Label(tracking_frame, text='Tracking Mode:').grid(row=0, column=0, sticky='w', padx=5, pady=5)
    tracking_mode_menu = tk.OptionMenu(tracking_frame, tracking_mode_var, 'predict', 'track', 'botsort')
    tracking_mode_menu.grid(row=0, column=1, sticky='w', padx=5)

    tk.Label(tracking_frame, text='IOU Threshold:').grid(row=0, column=2, sticky='w', padx=5, pady=5)
    iou_entry = tk.Entry(tracking_frame, textvariable=iou_var, width=10)
    iou_entry.grid(row=0, column=3, sticky='w', padx=5)

    # --- Bounding Box Combining Frame ---
    combine_frame = tk.LabelFrame(options_frame, text="Detection Combining", padx=5, pady=5)
    combine_frame.grid(row=3, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

    combine_check = tk.Checkbutton(combine_frame, text="Combine Overlapping BBoxes from different models",
                                   variable=combine_bboxes_var)
    combine_check.grid(row=0, column=0, columnspan=3, sticky='w', padx=5)

    tk.Label(combine_frame, text='Combine IoU Threshold:').grid(row=1, column=0, sticky='w', padx=5, pady=5)
    combine_iou_scale = tk.Scale(combine_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                                 variable=combine_iou_var, length=200)
    combine_iou_scale.grid(row=1, column=1, columnspan=2, sticky='w', padx=5)

    resize_frame = tk.LabelFrame(options_frame, text="YOLO Input Resize Mode", padx=5, pady=5)
    resize_frame.grid(row=4, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

    tk.Label(resize_frame, text='Input Size (pixels):').grid(row=0, column=0, sticky='w', padx=5, pady=5)
    resolutions = [str(x) for x in range(640, 3440, 32)]
    input_size_combo = ttk.Combobox(resize_frame, textvariable=input_size_var, values=resolutions, width=8)
    input_size_combo.grid(row=0, column=1, sticky='w', padx=5)
    input_size_combo.set(cfg.get('yolo_input_size', 640))

    scale_radio = tk.Radiobutton(resize_frame, text="Scale to Fit", variable=resize_mode_var, value="Scale")
    scale_radio.grid(row=1, column=0, sticky='w', padx=10)
    crop_radio = tk.Radiobutton(resize_frame, text="Center Crop", variable=resize_mode_var, value="Crop")
    crop_radio.grid(row=1, column=1, sticky='w', padx=10)

    dual_process_check = tk.Checkbutton(options_frame, text="Enable Dual Processing (Normal + Negative)",
                                        variable=dual_process_var)
    dual_process_check.grid(row=5, column=0, columnspan=2, sticky='w', padx=5, pady=5)
    display_id_check = tk.Checkbutton(options_frame, text="Display Class & ID on Box", variable=display_id_var)
    display_id_check.grid(row=6, column=0, sticky='w', padx=5, pady=5)
    show_crop_area_check = tk.Checkbutton(options_frame, text="Show Crop Area", variable=show_crop_area_var)
    show_crop_area_check.grid(row=6, column=1, sticky='w', padx=5, pady=5)
    show_fps_check = tk.Checkbutton(options_frame, text="Show FPS in Viewer", variable=show_fps_var)
    show_fps_check.grid(row=6, column=2, sticky='w', padx=5, pady=5)

    style_frame = tk.LabelFrame(options_frame, text="Drawing Styles", padx=5, pady=5)
    style_frame.grid(row=7, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

    tk.Label(style_frame, text='Box Thickness:').grid(row=0, column=0, sticky='w', padx=5, pady=2)
    thickness_entry = tk.Entry(style_frame, textvariable=thickness_var, width=10)
    thickness_entry.grid(row=0, column=1, sticky='w', padx=5)

    tk.Label(style_frame, text='Font Size:').grid(row=0, column=2, sticky='w', padx=5, pady=2)
    font_size_entry = tk.Entry(style_frame, textvariable=font_size_var, width=10)
    font_size_entry.grid(row=0, column=3, sticky='w', padx=5)

    override_frame = tk.LabelFrame(options_frame, text="ID Name Override", padx=5, pady=5)
    override_frame.grid(row=8, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

    override_enabled_check = tk.Checkbutton(override_frame, text="Enable ID Name Override",
                                            variable=override_enabled_var)
    override_enabled_check.grid(row=0, column=0, columnspan=2, sticky='w', padx=5)

    tk.Label(override_frame, text="Overrides (e.g., {'1': 'car', '2': 'person'}):").grid(row=1, column=0, sticky='w', padx=5,
                                                                               pady=2)
    override_text = tk.Text(override_frame, height=3, width=50)
    override_text.grid(row=2, column=0, columnspan=3, padx=5, pady=2)
    override_text.insert('1.0', cfg.get('override_text', ''))

    override_color_frame = tk.LabelFrame(options_frame, text="ID Color Override", padx=5, pady=5)
    override_color_frame.grid(row=9, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

    override_color_enabled_check = tk.Checkbutton(override_color_frame, text="Enable ID Color Override",
                                                  variable=override_color_enabled_var)
    override_color_enabled_check.grid(row=0, column=0, columnspan=2, sticky='w', padx=5)

    tk.Label(override_color_frame, text="Color Overrides (e.g., {'1': (0,0,255), '2': (255,0,0)})").grid(row=1, column=0,
                                                                                                   sticky='w', padx=5,
                                                                                                   pady=2)
    override_color_text = tk.Text(override_color_frame, height=3, width=50)
    override_color_text.grid(row=2, column=0, columnspan=3, padx=5, pady=2)
    override_color_text.insert('1.0', cfg.get('override_color_text', ''))

    def toggle_ui_state(running):
        state = 'disabled' if running else 'normal'
        # Iterate over all widgets and disable/enable them
        for child in root.winfo_children():
            # Keep control buttons and live-updatable widgets enabled
            if isinstance(child, tk.Frame):
                for widget in child.winfo_children():
                    # This check is a bit simplified, might need refinement
                    if isinstance(widget, (tk.Button, tk.Scale, tk.Checkbutton, tk.Radiobutton, tk.Entry, tk.Text, ttk.Combobox)):
                         try:
                            # Don't disable the main control buttons
                            if widget not in [run_stop_btn, apply_btn]:
                                widget.config(state=state)
                         except tk.TclError:
                             pass
        
        # Explicitly control the main buttons
        run_stop_btn.config(state='normal')
        apply_btn.config(state='normal' if running else 'disabled')


    def update_live_config():
        """Applies configuration changes to the running tracker without restarting."""
        global live_config
        live_config['conf_threshold'] = conf_var.get()
        live_config['dual_processing'] = dual_process_var.get()
        live_config['display_id'] = display_id_var.get()
        live_config['show_crop_area'] = show_crop_area_var.get()
        live_config['box_thickness'] = thickness_var.get()
        live_config['font_size'] = font_size_var.get()
        live_config['override_enabled'] = override_enabled_var.get()
        live_config['override_text'] = override_text.get('1.0', tk.END).strip()
        live_config['override_color_enabled'] = override_color_enabled_var.get()
        live_config['override_color_text'] = override_color_text.get('1.0', tk.END).strip()
        live_config['detections_per_second'] = detections_per_second_var.get()
        live_config['show_fps'] = show_fps_var.get()
        live_config['combine_bboxes'] = combine_bboxes_var.get()
        live_config['combine_iou_threshold'] = combine_iou_var.get()
        live_config['per_model_conf'] = per_model_conf_var.get()
        messagebox.showinfo("Info", "Live settings applied!")

    def on_run_stop():
        """Starts or stops the tracking thread."""
        global tracking_thread
        if tracking_thread and tracking_thread.is_alive():
            stop_event.set()
            run_stop_btn.config(text="Run Tracker")
            toggle_ui_state(False)
        else:
            if not model_list_data:
                messagebox.showerror('Error', 'Please add at least one model.')
                return

            cfg_to_save = {
                'models': model_list_data,
                'video_path': video_var.get().strip(),
                'conf_threshold': conf_var.get(),
                'use_webcam': use_webcam_var.get(),
                'webcam_id': webcam_id_var.get(),
                'dual_processing': dual_process_var.get(),
                'yolo_input_size': input_size_var.get(),
                'display_id': display_id_var.get(),
                'resize_mode': resize_mode_var.get(),
                'box_thickness': thickness_var.get(),
                'font_size': font_size_var.get(),
                'override_enabled': override_enabled_var.get(),
                'override_text': override_text.get('1.0', tk.END).strip(),
                'show_crop_area': show_crop_area_var.get(),
                'override_color_enabled': override_color_enabled_var.get(),
                'override_color_text': override_color_text.get('1.0', tk.END).strip(),
                'tracking_mode': tracking_mode_var.get(),
                'iou_threshold': iou_var.get(),
                'detections_per_second': detections_per_second_var.get(),
                'show_fps': show_fps_var.get(),
                'combine_bboxes': combine_bboxes_var.get(),
                'combine_iou_threshold': combine_iou_var.get(),
                'per_model_conf': per_model_conf_var.get(),
            }

            video_source = cfg_to_save['webcam_id'] if cfg_to_save['use_webcam'] else cfg_to_save['video_path']
            if not cfg_to_save['use_webcam'] and not os.path.exists(video_source):
                messagebox.showerror('Error', 'Video file not found.')
                return

            save_config(cfg_to_save)
            global live_config
            live_config = cfg_to_save.copy()

            stop_event.clear()
            tracking_thread = threading.Thread(target=run_tracking, args=(live_config,))
            tracking_thread.start()
            run_stop_btn.config(text="Stop Tracker")
            toggle_ui_state(True)

    # --- Control Buttons ---
    run_stop_btn = tk.Button(root, text='Run Tracker', command=on_run_stop, width=20, height=2)
    run_stop_btn.grid(row=3, column=0, pady=15)

    apply_btn = tk.Button(root, text='Apply Live Changes', command=update_live_config, width=20, height=2, state='disabled')
    apply_btn.grid(row=3, column=1, pady=15)

    def on_closing():
        if tracking_thread and tracking_thread.is_alive():
            stop_event.set()
            tracking_thread.join()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    update_model_listbox()
    root.mainloop()


# --- BBOX COMBINING LOGIC ---
def calculate_iou(box_a, box_b):
    """Calculates Intersection over Union for two bounding boxes."""
    # box format: [x1, y1, x2, y2]
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    iou = inter_area / float(box_a_area + box_b_area - inter_area + 1e-6)
    return iou


def combine_detections(detections, iou_threshold):
    """Combines overlapping bounding boxes using a greedy approach."""
    if not detections:
        return []

    # Sort by confidence
    detections.sort(key=lambda x: x['conf'], reverse=True)

    processed = [False] * len(detections)
    final_detections = []

    for i in range(len(detections)):
        if processed[i]:
            continue
        processed[i] = True

        current_group = [detections[i]]
        base_box = detections[i]['box']

        for j in range(i + 1, len(detections)):
            if processed[j]:
                continue

            # Check for same class and IoU threshold
            if detections[i]['cls'] == detections[j]['cls'] and calculate_iou(base_box,
                                                                              detections[j]['box']) > iou_threshold:
                processed[j] = True
                current_group.append(detections[j])

        if len(current_group) > 1:
            # Merge the boxes in the group by creating a union box
            all_boxes = np.array([d['box'] for d in current_group])
            union_box = [
                np.min(all_boxes[:, 0]),
                np.min(all_boxes[:, 1]),
                np.max(all_boxes[:, 2]),
                np.max(all_boxes[:, 3])
            ]

            # Use properties from the highest confidence box, but average the confidence
            primary_detection = current_group[0]
            avg_conf = np.mean([d['conf'] for d in current_group])

            final_detections.append({
                'box': union_box,
                'conf': avg_conf,
                'cls': primary_detection['cls'],
                'class_name': primary_detection['class_name'],
                'color': (255, 0, 255),  # Magenta for merged boxes
                'track_id': None,  # Merged boxes don't have a single track ID
                'is_merged': True
            })
        else:
            # Not merged, add the original detection
            detections[i]['is_merged'] = False
            final_detections.append(detections[i])

    return final_detections


# --- UNIFIED DRAWING FUNCTION ---
def draw_final_boxes(frame, detections, cfg, offset=(0, 0), override_names=None, override_colors=None):
    """Draws the final list of (potentially merged) bounding boxes, handling all styling and overrides."""
    if override_names is None: override_names = {}
    if override_colors is None: override_colors = {}

    box_thickness = cfg.get('box_thickness', 2)
    font_size = cfg.get('font_size', 0.6)
    offset_x, offset_y = offset

    override_enabled = cfg.get('override_enabled', False)
    override_color_enabled = cfg.get('override_color_enabled', False)
    display_id = cfg.get('display_id', True)

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det['box']]
        x1, y1, x2, y2 = x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y

        box_color = det['color']
        class_id = det['cls']
        track_id = det.get('track_id')

        # Color Override Logic
        if override_color_enabled:
            # Check for a specific override first by track_id, then by class_id
            effective_id = track_id if track_id is not None else class_id
            if effective_id in override_colors:
                box_color = override_colors[effective_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

        if not display_id:
            continue

        # Label Construction Logic
        label = ""
        class_name = det['class_name']
        
        # Name Override Logic
        if override_enabled:
            effective_id = track_id if track_id is not None else class_id
            class_name = override_names.get(effective_id, class_name)
        
        if det.get('is_merged', False):
            label = f"MERGED {class_name} {det['conf']:.2f}"
        else:
            parts = []
            if track_id is not None:
                parts.append(f'ID:{track_id}')
            parts.append(class_name)
            parts.append(f"{det['conf']:.2f}")
            label = ' '.join(parts)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, box_thickness)
        # Ensure the label background doesn't go off-screen
        label_y1 = y1 - h - 5
        cv2.rectangle(frame, (x1, label_y1), (x1 + w, y1), box_color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), box_thickness)


def run_tracking(cfg):
    """Main function to run the video tracking loop."""
    global live_config
    loaded_models = []
    for model_config in cfg.get('models', []):
        try:
            # Use a dictionary to store model and its properties for easier access
            model_info = {
                'model': YOLO(model_config['path']).to('cuda'),
                'color': model_config['color'],
                'conf': model_config.get('conf', cfg.get('conf_threshold'))
            }
            # Add a unique tracker instance for each model if in tracking mode
            if cfg.get('tracking_mode', 'predict') != 'predict':
                model_info['tracker'] = YOLO(model_config['path']).to('cuda')  # Separate instance for tracking
            loaded_models.append(model_info)
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load YOLO model: {model_config['path']}\n{e}")
            return

    video_source = int(cfg['webcam_id']) if cfg['use_webcam'] else cfg['video_path']
    cap = cv2.VideoCapture(video_source)

    if cfg['use_webcam']:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

    if not cap.isOpened():
        messagebox.showerror("Video Error", f"Could not open video source: {video_source}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0: video_fps = 30

    frame_counter = 0
    fps_display, fps_frame_count, fps_start_time = "0", 0, time.time()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, VIEWER_WIDTH, VIEWER_HEIGHT)

    # This will hold the detections from the last processed frame
    last_detections = []

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break

        frame_counter += 1
        current_cfg = live_config.copy()

        # Parse override configurations once per frame
        override_names, override_colors = {}, {}
        try:
            if current_cfg.get('override_enabled', False) and current_cfg.get('override_text', ''):
                override_names = ast.literal_eval(current_cfg['override_text'])
            if current_cfg.get('override_color_enabled', False) and current_cfg.get('override_color_text', ''):
                override_colors = ast.literal_eval(current_cfg['override_color_text'])
        except Exception as e:
            print(f"Error parsing overrides: {e}")  # Non-critical error

        dps = current_cfg.get('detections_per_second', 10)
        detect_interval = max(1, round(video_fps / dps)) if dps > 0 else 1

        # Determine if this is a detection frame based on mode
        tracking_mode = current_cfg.get('tracking_mode', 'predict')
        is_detection_frame = (frame_counter % detect_interval == 0) or (tracking_mode != 'predict')

        # Prepare input frame and crop if necessary
        input_frame = frame.copy()
        crop_offset = (0, 0)
        if current_cfg['resize_mode'] == 'Crop':
            h, w, _ = input_frame.shape
            size = current_cfg['yolo_input_size']
            if h > size and w > size:
                start_x, start_y = (w - size) // 2, (h - size) // 2
                input_frame = input_frame[start_y:start_y + size, start_x:start_x + size]
                crop_offset = (start_x, start_y)
                if current_cfg.get('show_crop_area', False):
                    cv2.rectangle(frame, (start_x, start_y), (start_x + size, start_y + size), (0, 0, 255), 2)

        if is_detection_frame:
            all_detections_this_frame = []
            for item in loaded_models:
                model = item['model']
                conf = item['conf'] if current_cfg.get('per_model_conf') else current_cfg['conf_threshold']
                common_args = {'source': input_frame, 'conf': conf, 'iou': current_cfg.get('iou_threshold', 0.5),
                               'device': 'cuda', 'imgsz': current_cfg['yolo_input_size'], 'verbose': False}

                results_list = []
                if tracking_mode == 'predict':
                    results = model.predict(**common_args)
                    results_list.append(results)
                    if current_cfg['dual_processing']:
                        common_args['source'] = cv2.bitwise_not(input_frame)
                        results_list.append(model.predict(**common_args))
                else:  # Tracking modes
                    tracker_file = "botsort.yaml" if tracking_mode == 'botsort' else "bytetrack.yaml"
                    results = model.track(**common_args, persist=True, tracker=tracker_file)
                    results_list.append(results)
                    if current_cfg['dual_processing']:
                        common_args['source'] = cv2.bitwise_not(input_frame)
                        results_list.append(model.track(**common_args, persist=True, tracker=tracker_file))

                # Process results from normal and negative images
                for results in results_list:
                    if results and results[0].boxes:
                        boxes = results[0].boxes
                        class_names = results[0].names
                        track_ids = boxes.id.cpu().numpy().astype(int) if hasattr(boxes,
                                                                                  'id') and boxes.id is not None else [None] * len(
                            boxes.xyxy)
                        for j in range(len(boxes.xyxy)):
                            all_detections_this_frame.append({
                                'box': boxes.xyxy.cpu().numpy()[j],
                                'conf': boxes.conf.cpu().numpy()[j],
                                'cls': int(boxes.cls.cpu().numpy()[j]),
                                'class_name': class_names.get(int(boxes.cls.cpu().numpy()[j]), 'unknown'),
                                'color': item['color'],
                                'track_id': track_ids[j]
                            })

            # Update last_detections with the new findings for this frame
            last_detections = all_detections_this_frame

        # In 'predict' mode on a non-detection frame, clear old boxes
        elif tracking_mode == 'predict' and not is_detection_frame:
            last_detections = []

        # --- Combine and Draw ---
        final_boxes_to_draw = []
        if current_cfg.get('combine_bboxes', False) and last_detections:
            final_boxes_to_draw = combine_detections(last_detections,
                                                     current_cfg.get('combine_iou_threshold', 0.6))
        else:
            for det in last_detections:
                det['is_merged'] = False
            final_boxes_to_draw = last_detections

        draw_final_boxes(frame, final_boxes_to_draw, current_cfg, crop_offset, override_names, override_colors)

        # --- FPS Calculation and Display ---
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps_display = str(fps_frame_count)
            fps_frame_count = 0
            fps_start_time = time.time()
        if current_cfg.get('show_fps', True):
            cv2.putText(frame, f"FPS: {fps_display}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Display the final frame
        display_frame = cv2.resize(frame, (VIEWER_WIDTH, VIEWER_HEIGHT))
        cv2.imshow(WINDOW_NAME, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_event.clear()

    # This part might need to run on the main thread, but let's keep it simple
    print("Tracking stopped. UI should be re-enabled.")
    if 'root' in locals() and root.winfo_exists():
       toggle_ui_state(False)
       run_stop_btn.config(text="Run Tracker")


if __name__ == '__main__':
    build_ui()