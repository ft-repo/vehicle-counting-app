#!/usr/bin/env python3
"""
vehicle_counter.py
============================================================
Real-time vehicle counting and classification using YOLOv4/v4-tiny
and OpenCV DNN backend.  Mac/Linux CPU (Metal GPU optional).

Port of old_win_code/vehicle_counter.cpp with Windows IPC removed.
All scene_config.json fields (ROI, lanes, directions) are preserved.

Usage:
  python vehicle_counter.py <video_or_rtsp> [options]

Options:
  --cfg     <path>   YOLO .cfg file
  --weights <path>   YOLO .weights file
  --names   <path>   Class names file
  --config  <path>   scene_config.json (ROI + lanes)
  --size    <int>    YOLO input size (default 416)
  --conf    <float>  Confidence threshold (default 0.35)
  --nms     <float>  NMS threshold (default 0.40)
  --csv     <path>   CSV output path (default vehicle_counts.csv)
  --out     <path>   Save annotated video to file
  --nowin            Headless mode (no display)
  --cpu              Force CPU backend
============================================================
"""

import argparse
import csv
import json
import math
import os
import platform
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# ============================================================
#  Windows DPI helper
# ============================================================
def _win32_client_size(title: str):
    """Return (w, h) of a named OpenCV window via Win32 GetClientRect.

    Uses physical pixels regardless of DPI scaling mode — same coordinate
    space as WM_MOUSEMOVE events — so the computed scale is always correct.
    Returns None if the window can't be found or Win32 is unavailable.
    """
    try:
        import ctypes
        import ctypes.wintypes as wt
        hwnd = ctypes.windll.user32.FindWindowW(None, title)
        if not hwnd:
            return None
        rc = wt.RECT()
        if ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rc)):
            w = rc.right  - rc.left
            h = rc.bottom - rc.top
            if w > 10 and h > 10:
                return (w, h)
    except Exception:
        pass
    return None


# ============================================================
#  Constants
# ============================================================
IOU_MATCH_THRESH = 0.30
MAX_LOST_FRAMES  = 30
MIN_HIT_STREAK   = 2

# Classes to exclude from counting and display
SKIP_CLASSES: set = {"agri_truck"}

CLASS_COLORS = [
    ( 51, 204, 255),  # person
    (255,  51,  51),  # car
    (255, 204,  51),  # bike
    ( 51, 255, 153),  # truck
    (204,  51, 255),  # bus
    ( 51, 102, 255),  # taxi
    (153, 255,  51),  # pickup
    (255, 153, 204),  # trailer
    (153, 204, 255),
    (255, 255,  51),
    (204, 255, 153),
    ( 51, 255, 255),
    (255,  51, 204),
]

def class_color(cls_id: int) -> Tuple[int, int, int]:
    return CLASS_COLORS[cls_id % len(CLASS_COLORS)]


# ============================================================
#  Configuration
# ============================================================
def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    h = hex_str.lstrip('#')
    if len(h) < 6:
        return (0, 255, 0)
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)  # OpenCV BGR


@dataclass
class LaneCfg:
    id:        int
    name:      str        = "Lane"
    enabled:   bool       = True
    color_hex: str        = "#00FF00"
    x1: int = 0; y1: int = 0; x2: int = 0; y2: int = 0
    direction: str        = "both"
    arrow_in:  str        = "top_to_bottom"
    count_in:  bool       = True
    count_out: bool       = True
    # runtime counters (not serialized)
    cross_in:    int = 0
    cross_out:   int = 0
    flash_frames: int = 0

    def color(self) -> Tuple[int, int, int]:
        return hex_to_bgr(self.color_hex)


@dataclass
class SceneCfg:
    roi_enabled: bool             = False
    roi_pts:     List[Tuple[int,int]] = field(default_factory=list)
    lanes:       List[LaneCfg]   = field(default_factory=list)
    path:        str              = ""

    @classmethod
    def load(cls, file_path: str) -> "SceneCfg":
        cfg = cls(path=file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)

        roi = data.get('roi', {})
        cfg.roi_enabled = bool(roi.get('enabled', False))
        cfg.roi_pts = [tuple(p) for p in roi.get('points', [])]

        for ld in data.get('lanes', []):
            line = ld.get('line', {})
            lane = LaneCfg(
                id        = ld.get('id', 0),
                name      = ld.get('name', 'Lane'),
                enabled   = bool(ld.get('enabled', True)),
                color_hex = ld.get('color', '#00FF00'),
                x1 = int(line.get('x1', 0)),
                y1 = int(line.get('y1', 0)),
                x2 = int(line.get('x2', 0)),
                y2 = int(line.get('y2', 0)),
                direction = ld.get('direction', 'both'),
                arrow_in  = ld.get('arrow_in', 'top_to_bottom'),
                count_in  = bool(ld.get('count_in', True)),
                count_out = bool(ld.get('count_out', True)),
            )
            cfg.lanes.append(lane)

        print(f"[CFG] Loaded {len(cfg.lanes)} lanes, {len(cfg.roi_pts)} ROI points from {file_path}")
        return cfg

    def save(self):
        if not self.path:
            return
        data = {
            "version": "1.0",
            "roi": {
                "enabled": self.roi_enabled,
                "points": [list(p) for p in self.roi_pts]
            },
            "lanes": [
                {
                    "id":        l.id,
                    "name":      l.name,
                    "enabled":   l.enabled,
                    "color":     l.color_hex,
                    "line":      {"x1": l.x1, "y1": l.y1, "x2": l.x2, "y2": l.y2},
                    "direction": l.direction,
                    "arrow_in":  l.arrow_in,
                    "count_in":  l.count_in,
                    "count_out": l.count_out,
                }
                for l in self.lanes
            ]
        }
        # Preserve camera/yolo sections from existing file
        try:
            with open(self.path, 'r') as f:
                existing = json.load(f)
            for k in ('camera', 'yolo'):
                if k in existing:
                    data[k] = existing[k]
        except Exception:
            pass
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[CFG] Saved to {self.path}")


# ============================================================
#  Lane crossing detection
# ============================================================
def lane_crossed(prev: Tuple[float,float], curr: Tuple[float,float],
                 lane: LaneCfg) -> int:
    """Returns +1 (in), -1 (out), 0 (no cross).
    Checks only the finite segment [P1,P2]."""
    dx = lane.x2 - lane.x1
    dy = lane.y2 - lane.y1

    def side_of(pt):
        return dx * (pt[1] - lane.y1) - dy * (pt[0] - lane.x1)

    prev_c = side_of(prev)
    curr_c = side_of(curr)

    if prev_c == 0 and curr_c == 0:
        return 0
    if (prev_c >= 0) == (curr_c >= 0):
        return 0

    # Interpolate crossing point on prev→curr
    t  = prev_c / (prev_c - curr_c)
    cx = prev[0] + t * (curr[0] - prev[0])
    cy = prev[1] + t * (curr[1] - prev[1])

    # Project onto lane segment
    len_sq = dx*dx + dy*dy
    if len_sq < 1:
        return 0
    s = ((cx - lane.x1) * dx + (cy - lane.y1) * dy) / len_sq
    if s < 0 or s > 1:
        return 0

    neg_to_pos = (prev_c < 0 and curr_c > 0)
    arrow_neg_to_pos = lane.arrow_in in ("top_to_bottom", "left_to_right")
    is_in = neg_to_pos if arrow_neg_to_pos else not neg_to_pos
    return +1 if is_in else -1


# ============================================================
#  Detection / Track
# ============================================================
@dataclass
class Detection:
    rect:       Tuple[int,int,int,int]  # x, y, w, h
    class_id:   int   = 0
    confidence: float = 0.0

    def centroid(self) -> Tuple[float, float]:
        x, y, w, h = self.rect
        return (x + w * 0.5, y + h * 0.5)


def rect_iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = min(ax + aw, bx + bw) - ix
    ih = min(ay + ah, by + bh) - iy
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class Track:
    _next_id = 1

    def __init__(self, det: Detection):
        self.id          = Track._next_id
        Track._next_id  += 1
        self.rect        = det.rect
        self.center      = det.centroid()
        self.prev_center = det.centroid()
        self.class_id    = det.class_id
        self.confidence  = det.confidence
        self.hit_streak  = 1
        self.lost_frames = 0
        self.counted_lanes: Set[int] = set()

    def update(self, det: Detection):
        self.prev_center = self.center
        self.rect        = det.rect
        self.center      = det.centroid()
        self.class_id    = det.class_id
        self.confidence  = det.confidence
        self.hit_streak += 1
        self.lost_frames = 0

    def mark_lost(self):
        self.prev_center = self.center
        self.lost_frames += 1
        self.hit_streak  = 0


class CentroidTracker:
    def __init__(self, max_lost: int = MAX_LOST_FRAMES,
                 iou_thresh: float = IOU_MATCH_THRESH):
        self.tracks: List[Track] = []
        self.max_lost  = max_lost
        self.iou_thresh = iou_thresh

    def update(self, dets: List[Detection]) -> List[Track]:
        det_used = [False] * len(dets)
        trk_used = [False] * len(self.tracks)

        for ti, trk in enumerate(self.tracks):
            best_iou = self.iou_thresh
            best_di  = -1
            for di, det in enumerate(dets):
                if det_used[di]:
                    continue
                score = rect_iou(trk.rect, det.rect)
                if score > best_iou:
                    best_iou = score
                    best_di  = di
            if best_di >= 0:
                self.tracks[ti].update(dets[best_di])
                det_used[best_di] = True
                trk_used[ti]      = True

        for ti, trk in enumerate(self.tracks):
            if not trk_used[ti]:
                self.tracks[ti].mark_lost()

        for di, det in enumerate(dets):
            if not det_used[di]:
                self.tracks.append(Track(det))

        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost]
        return self.tracks

    def reset(self):
        self.tracks.clear()


# ============================================================
#  YOLO Detector
# ============================================================
class YoloDetector:
    def __init__(self):
        self.net         = None
        self.out_names   = []
        self.class_names: List[str] = []
        self.conf_thresh = 0.35
        self.nms_thresh  = 0.40
        self.net_size    = 416
        self.model_type  = 'darknet'   # 'darknet' or 'yolov8'

    def load(self, cfg_file: str = '', weights_file: str = '', names_file: str = '',
             use_gpu: bool = False, onnx_file: str = '', force_gpu: bool = False) -> bool:
        # Load class names
        with open(names_file, 'r') as f:
            self.class_names = [l.strip() for l in f if l.strip()]
        print(f"[INFO] Loaded {len(self.class_names)} class names.")

        # Load network
        try:
            if onnx_file:
                self.net        = cv2.dnn.readNetFromONNX(onnx_file)
                self.model_type = 'yolov8'
                # net_size stays as whatever was set externally (from --size or scene_config input_size)
                print(f"[INFO] Loaded YOLOv8 ONNX model: {onnx_file} at {self.net_size}px")
            else:
                self.net        = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
                self.model_type = 'darknet'
        except cv2.error as e:
            print(f"[ERROR] OpenCV DNN load failed: {e}", file=sys.stderr)
            return False

        if self.net.empty():
            print("[ERROR] Network is empty.", file=sys.stderr)
            return False

        # Backend selection
        if use_gpu:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                # Quick probe to confirm CUDA actually works
                dummy = np.zeros((self.net_size, self.net_size, 3), np.uint8)
                blob  = cv2.dnn.blobFromImage(dummy, 1/255.0,
                            (self.net_size, self.net_size), swapRB=True, crop=False)
                self.net.setInput(blob)
                self.net.forward(self.net.getUnconnectedOutLayersNames())
                print("[INFO] Backend: CUDA GPU")
            except Exception as e:
                if force_gpu:
                    print(f"[ERROR] GPU requested but CUDA init failed: {e}")
                    print("[ERROR] Install CUDA-enabled OpenCV or choose CPU option.")
                    return False
                print(f"[WARN] GPU init failed ({e}), falling back to CPU.")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("[INFO] Backend: OpenCV DNN CPU")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("[INFO] Backend: OpenCV DNN CPU")

        self.out_names = self.net.getUnconnectedOutLayersNames()
        return True

    def class_name(self, cls_id: int) -> str:
        if 0 <= cls_id < len(self.class_names):
            return self.class_names[cls_id]
        return "unknown"

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0,
                    (self.net_size, self.net_size),
                    swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_names)
        return self._post_process(w, h, outs)

    def _post_process(self, W: int, H: int, outs) -> List[Detection]:
        if self.model_type == 'yolov8':
            return self._post_process_yolov8(W, H, outs)

        # --- YOLOv4 darknet output ---
        # Each 'out' is shape (N, 5+classes), values normalized 0-1
        # row = [cx, cy, w, h, obj_conf, class0, class1, ...]
        class_ids, confs, boxes = [], [], []

        for out in outs:
            for row in out:
                scores = row[5:]
                cls_id = int(np.argmax(scores))
                conf   = float(row[4]) * float(scores[cls_id])
                if conf < self.conf_thresh:
                    continue
                cx = int(row[0] * W)
                cy = int(row[1] * H)
                bw = int(row[2] * W)
                bh = int(row[3] * H)
                x  = max(0, cx - bw // 2)
                y  = max(0, cy - bh // 2)
                bw = min(bw, W - x)
                bh = min(bh, H - y)
                class_ids.append(cls_id)
                confs.append(conf)
                boxes.append([x, y, bw, bh])

        indices = cv2.dnn.NMSBoxes(boxes, confs, self.conf_thresh, self.nms_thresh)
        result  = []
        if len(indices) > 0:
            for i in indices.flatten():
                result.append(Detection(
                    rect       = tuple(boxes[i]),
                    class_id   = class_ids[i],
                    confidence = confs[i],
                ))
        return result

    def _post_process_yolov8(self, W: int, H: int, outs) -> List[Detection]:
        # YOLOv8 ONNX output shape: (1, 4+nc, 8400)
        # Transpose to (8400, 4+nc) so each row = 1 candidate box
        # row = [cx, cy, w, h, class0_score, class1_score, ...]
        # Coordinates are in 640px space — scale to original frame size
        predictions = outs[0][0].T
        scale_x = W / self.net_size
        scale_y = H / self.net_size
        class_ids, confs, boxes = [], [], []

        for row in predictions:
            scores = row[4:]
            cls_id = int(np.argmax(scores))
            conf   = float(scores[cls_id])
            if conf < self.conf_thresh:
                continue
            cx = row[0] * scale_x
            cy = row[1] * scale_y
            bw = row[2] * scale_x
            bh = row[3] * scale_y
            x  = max(0, int(cx - bw / 2))
            y  = max(0, int(cy - bh / 2))
            bw = min(int(bw), W - x)
            bh = min(int(bh), H - y)
            class_ids.append(cls_id)
            confs.append(conf)
            boxes.append([x, y, bw, bh])

        indices = cv2.dnn.NMSBoxes(boxes, confs, self.conf_thresh, self.nms_thresh)
        result  = []
        if len(indices) > 0:
            for i in indices.flatten():
                result.append(Detection(
                    rect       = tuple(boxes[i]),
                    class_id   = class_ids[i],
                    confidence = confs[i],
                ))
        return result


# ============================================================
#  UI State  (Edit / DrawRoi / AddLane modes)
# ============================================================
class UIMode(Enum):
    NONE         = auto()
    EDIT_SCENE   = auto()
    DRAW_ROI     = auto()
    ADD_LANE     = auto()
    CAMERA_INPUT = auto()

class DragTarget(Enum):
    NONE      = auto()
    ROI_PT    = auto()
    LANE_EP   = auto()
    LANE_LINE = auto()


HIT_R_SCREEN  = 28   # screen-space hit radius for ROI/lane endpoint clicks
LINE_D_SCREEN = 20   # screen-space hit distance for lane body dragging


class UIState:
    def __init__(self):
        self.mode         = UIMode.NONE
        self.roi_draft: List[Tuple[int,int]] = []
        self.drag_target  = DragTarget.NONE
        self.drag_roi_idx   = -1
        self.drag_lane_idx  = -1
        self.drag_endpoint  = -1   # 0=P1, 1=P2
        self.drag_offset    = (0, 0)
        self.hover_roi      = -1
        self.hover_roi_edge = -1
        self.hover_lane     = -1
        self.hover_end      = -1
        self.selected_lane  = -1
        self.selected_roi_pt = -1
        self.add_lane_dragging = False
        self.add_lane_start  = (0, 0)
        self.add_lane_current = (0, 0)
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.need_save = False
        self.cam_input_buf: str = ""
        self.cam_input_error: bool = False
        self.cam_presets: List[str] = []
        self.cam_preset_idx: int = -1
        self.cam_rtsp_template: str = "rtsp://root:pass@{host}/axis-media/media.amp"

    def hit_r(self) -> int:
        return max(6, int(HIT_R_SCREEN * self.scale_x))

    def line_d(self) -> int:
        return max(4, int(LINE_D_SCREEN * self.scale_x))


def _pt_dist(a, b) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])


def _line_dist(p, a, b) -> float:
    dx, dy = b[0]-a[0], b[1]-a[1]
    len2 = dx*dx + dy*dy
    if len2 < 1:
        return _pt_dist(p, a)
    t = max(0.0, min(1.0, ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / len2))
    px, py = a[0]+t*dx, a[1]+t*dy
    return math.hypot(p[0]-px, p[1]-py)


def _edge_mid(pts, i):
    j = (i + 1) % len(pts)
    return ((pts[i][0]+pts[j][0])//2, (pts[i][1]+pts[j][1])//2)


def _clamp_pt(p, W, H, pad=10):
    return (max(pad, min(W-pad, p[0])), max(pad, min(H-pad, p[1])))


def make_mouse_callback(ui: UIState, cfg: SceneCfg, frame_wh):
    FW, FH = frame_wh[0], frame_wh[1]

    def on_mouse(event, x, y, flags, param):
        nonlocal FW, FH
        fx = int(x * ui.scale_x)
        fy = int(y * ui.scale_y)
        fp = _clamp_pt((fx, fy), FW, FH)
        HIT_R  = ui.hit_r()
        LINE_D = ui.line_d()

        if event == cv2.EVENT_LBUTTONDOWN:
            if ui.mode == UIMode.DRAW_ROI:
                cfg.roi_pts.append(fp)
                return
            if ui.mode == UIMode.ADD_LANE:
                ui.add_lane_dragging = True
                ui.add_lane_start    = fp
                ui.add_lane_current  = fp
                return
            if ui.mode == UIMode.EDIT_SCENE:
                # Priority 1: ROI vertex
                for i, pt in enumerate(cfg.roi_pts):
                    if _pt_dist(fp, pt) < HIT_R:
                        ui.drag_target    = DragTarget.ROI_PT
                        ui.drag_roi_idx   = i
                        ui.selected_roi_pt = i
                        ui.selected_lane  = -1
                        return
                # Priority 2: ROI edge midpoint → insert
                if ui.hover_roi_edge >= 0 and len(cfg.roi_pts) >= 3:
                    ei  = ui.hover_roi_edge
                    mid = _edge_mid(cfg.roi_pts, ei)
                    cfg.roi_pts.insert(ei + 1, mid)
                    ui.drag_target    = DragTarget.ROI_PT
                    ui.drag_roi_idx   = ei + 1
                    ui.hover_roi_edge = -1
                    ui.need_save      = True
                    return
                # Priority 3: Lane endpoint
                for li, lane in enumerate(cfg.lanes):
                    p1, p2 = (lane.x1, lane.y1), (lane.x2, lane.y2)
                    if _pt_dist(fp, p1) < HIT_R:
                        ui.drag_target   = DragTarget.LANE_EP
                        ui.drag_lane_idx = li; ui.drag_endpoint = 0
                        ui.selected_lane = li
                        return
                    if _pt_dist(fp, p2) < HIT_R:
                        ui.drag_target   = DragTarget.LANE_EP
                        ui.drag_lane_idx = li; ui.drag_endpoint = 1
                        ui.selected_lane = li
                        return
                # Priority 4: Lane body
                for li, lane in enumerate(cfg.lanes):
                    p1, p2 = (lane.x1, lane.y1), (lane.x2, lane.y2)
                    if _line_dist(fp, p1, p2) < LINE_D:
                        ui.drag_target   = DragTarget.LANE_LINE
                        ui.drag_lane_idx = li
                        ui.drag_offset   = fp
                        ui.selected_lane = li
                        return
                ui.selected_lane   = -1
                ui.selected_roi_pt = -1

        elif event == cv2.EVENT_MOUSEMOVE:
            if ui.mode == UIMode.ADD_LANE:
                ui.add_lane_current = fp
                return
            if ui.mode == UIMode.EDIT_SCENE:
                if ui.drag_target == DragTarget.ROI_PT and 0 <= ui.drag_roi_idx < len(cfg.roi_pts):
                    cfg.roi_pts[ui.drag_roi_idx] = fp
                    return
                if ui.drag_target == DragTarget.LANE_EP and 0 <= ui.drag_lane_idx < len(cfg.lanes):
                    l = cfg.lanes[ui.drag_lane_idx]
                    if ui.drag_endpoint == 0:
                        l.x1, l.y1 = fx, fy
                    else:
                        l.x2, l.y2 = fx, fy
                    return
                if ui.drag_target == DragTarget.LANE_LINE and 0 <= ui.drag_lane_idx < len(cfg.lanes):
                    l   = cfg.lanes[ui.drag_lane_idx]
                    ddx = fp[0] - ui.drag_offset[0]
                    ddy = fp[1] - ui.drag_offset[1]
                    np1 = _clamp_pt((l.x1+ddx, l.y1+ddy), FW, FH)
                    np2 = _clamp_pt((l.x2+ddx, l.y2+ddy), FW, FH)
                    l.x1, l.y1 = np1; l.x2, l.y2 = np2
                    ui.drag_offset = fp
                    return
                # Update hover
                ui.hover_roi = ui.hover_roi_edge = ui.hover_lane = ui.hover_end = -1
                for i, pt in enumerate(cfg.roi_pts):
                    if _pt_dist(fp, pt) < HIT_R:
                        ui.hover_roi = i; return
                for i in range(len(cfg.roi_pts)):
                    mid = _edge_mid(cfg.roi_pts, i)
                    if _pt_dist(fp, mid) < HIT_R:
                        ui.hover_roi_edge = i; return
                for li, lane in enumerate(cfg.lanes):
                    p1, p2 = (lane.x1, lane.y1), (lane.x2, lane.y2)
                    if _pt_dist(fp, p1) < HIT_R:
                        ui.hover_lane = li; ui.hover_end = 0; return
                    if _pt_dist(fp, p2) < HIT_R:
                        ui.hover_lane = li; ui.hover_end = 1; return
                    if _line_dist(fp, p1, p2) < LINE_D:
                        ui.hover_lane = li; ui.hover_end = 2; return

        elif event == cv2.EVENT_LBUTTONUP:
            if ui.mode == UIMode.ADD_LANE and ui.add_lane_dragging:
                sx, sy = ui.add_lane_start
                ex, ey = ui.add_lane_current
                if _pt_dist(ui.add_lane_start, ui.add_lane_current) > 10:
                    new_id = max((l.id for l in cfg.lanes), default=0) + 1
                    COLORS = ["#00FF00","#FF8800","#00AAFF","#FF00FF","#FFFF00"]
                    col = COLORS[(new_id - 1) % len(COLORS)]
                    cfg.lanes.append(LaneCfg(
                        id=new_id, name=f"Lane {new_id}", color_hex=col,
                        x1=sx, y1=sy, x2=ex, y2=ey
                    ))
                    ui.need_save = True
                ui.add_lane_dragging = False
            if ui.drag_target != DragTarget.NONE:
                ui.need_save   = True
            ui.drag_target   = DragTarget.NONE
            ui.drag_roi_idx  = -1
            ui.drag_lane_idx = -1

        elif event == cv2.EVENT_RBUTTONDOWN:
            if ui.mode == UIMode.DRAW_ROI:
                if len(cfg.roi_pts) >= 3:
                    cfg.roi_enabled = True
                ui.mode    = UIMode.NONE
                ui.roi_draft = []
                ui.need_save = True
                return
            if ui.mode == UIMode.EDIT_SCENE:
                for i, pt in enumerate(cfg.roi_pts):
                    if _pt_dist(fp, pt) < HIT_R:
                        cfg.roi_pts.pop(i)
                        ui.need_save = True
                        return
                for li, lane in enumerate(cfg.lanes):
                    p1, p2 = (lane.x1, lane.y1), (lane.x2, lane.y2)
                    if _pt_dist(fp, p1) < HIT_R or _pt_dist(fp, p2) < HIT_R or \
                       _line_dist(fp, p1, p2) < LINE_D:
                        cfg.lanes.pop(li)
                        ui.need_save = True
                        return

    return on_mouse


# ============================================================
#  Draw helpers
# ============================================================
def draw_scene_overlay(frame: np.ndarray, cfg: SceneCfg, ui: UIState):
    edit = (ui.mode == UIMode.EDIT_SCENE)

    # ROI draft
    for i, pt in enumerate(ui.roi_draft):
        cv2.circle(frame, pt, 5, (0,255,255), -1, cv2.LINE_AA)
        if i > 0:
            cv2.line(frame, ui.roi_draft[i-1], pt, (0,255,255), 1, cv2.LINE_AA)
    if len(ui.roi_draft) >= 3:
        cv2.line(frame, ui.roi_draft[-1], ui.roi_draft[0], (0,200,200), 1, cv2.LINE_AA)

    # ROI polygon
    if cfg.roi_enabled and len(cfg.roi_pts) >= 3:
        pts = [tuple(p) for p in cfg.roi_pts]
        for i in range(len(pts)):
            a, b = pts[i], pts[(i+1) % len(pts)]
            if edit:
                cv2.line(frame, a, b, (0,220,220), 2, cv2.LINE_AA)
            else:
                # dashed
                length = max(1, int(math.hypot(b[0]-a[0], b[1]-a[1])))
                segs   = max(1, length // 12)
                for s in range(segs):
                    if s % 2 == 1:
                        continue
                    pa = (a[0]+(b[0]-a[0])*s//segs, a[1]+(b[1]-a[1])*s//segs)
                    pb = (a[0]+(b[0]-a[0])*(s+1)//segs, a[1]+(b[1]-a[1])*(s+1)//segs)
                    cv2.line(frame, pa, pb, (0,220,220), 2, cv2.LINE_AA)

        vtx_r = 10 if edit else 7
        for i, pt in enumerate(pts):
            dragged = (ui.drag_target == DragTarget.ROI_PT and ui.drag_roi_idx == i)
            if dragged:            col = (255,255,255)
            elif ui.hover_roi == i: col = (255,255,150)
            else:                  col = (0,220,220)
            cv2.circle(frame, pt, vtx_r, col, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, vtx_r, (0,0,0), 1, cv2.LINE_AA)

        if edit:
            for i in range(len(pts)):
                mid = _edge_mid(pts, i)
                hov = (ui.hover_roi_edge == i)
                mc  = (255,255,100) if hov else (100,200,200)
                r   = 8 if hov else 5
                cv2.circle(frame, mid, r, mc, 1, cv2.LINE_AA)
                cv2.line(frame, (mid[0], mid[1]-r+1), (mid[0], mid[1]+r-1), mc, 1)
                cv2.line(frame, (mid[0]-r+1, mid[1]), (mid[0]+r-1, mid[1]), mc, 1)

    # Lanes
    ep_r = 10 if edit else 7
    for li, lane in enumerate(cfg.lanes):
        if not lane.enabled:
            continue
        p1 = (lane.x1, lane.y1)
        p2 = (lane.x2, lane.y2)
        col = lane.color()
        flash = lane.flash_frames > 0
        draw_col = (255,255,255) if flash else col
        line_w   = 5 if flash else 2

        if ui.selected_lane == li:
            cv2.line(frame, p1, p2, (255,255,255), 6, cv2.LINE_AA)
        cv2.line(frame, p1, p2, draw_col, line_w, cv2.LINE_AA)

        # Arrow
        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        dx  = p2[0]-p1[0]; dy = p2[1]-p1[1]
        length = math.hypot(dx, dy)
        if length > 1:
            if lane.arrow_in == "top_to_bottom":    nx, ny = 0, 1
            elif lane.arrow_in == "bottom_to_top":  nx, ny = 0, -1
            elif lane.arrow_in == "left_to_right":  nx, ny = 1, 0
            elif lane.arrow_in == "right_to_left":  nx, ny = -1, 0
            else:                                    nx, ny = -dy/length, dx/length
            tip = (int(mid[0]+nx*22), int(mid[1]+ny*22))
            cv2.arrowedLine(frame, mid, tip, draw_col, 2, cv2.LINE_AA, 0, 0.4)

        if edit:
            ep1_col = (255,255,255) if (ui.hover_lane==li and ui.hover_end==0) else col
            ep2_col = (255,255,255) if (ui.hover_lane==li and ui.hover_end==1) else col
            cv2.circle(frame, p1, ep_r, ep1_col, -1, cv2.LINE_AA)
            cv2.circle(frame, p1, ep_r, (0,0,0), 1, cv2.LINE_AA)
            cv2.circle(frame, p2, ep_r, ep2_col, -1, cv2.LINE_AA)
            cv2.circle(frame, p2, ep_r, (0,0,0), 1, cv2.LINE_AA)

        # Lane name
        tx, ty = mid[0]+6, mid[1]-8
        cv2.putText(frame, lane.name, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, lane.name, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_col, 1, cv2.LINE_AA)

        if flash:
            big = str(lane.cross_in)
            (bw, bh), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            bp = (mid[0]-bw//2, mid[1]-14)
            cv2.putText(frame, big, bp, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(frame, big, bp, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,100), 3, cv2.LINE_AA)

    # Add-lane preview
    if ui.mode == UIMode.ADD_LANE and ui.add_lane_dragging:
        cv2.line(frame, ui.add_lane_start, ui.add_lane_current, (0,255,100), 2, cv2.LINE_AA)
        cv2.circle(frame, ui.add_lane_start,   8, (0,255,100), -1, cv2.LINE_AA)
        cv2.circle(frame, ui.add_lane_current, 8, (0,255,100), -1, cv2.LINE_AA)


def _dir_short(arrow_in: str) -> str:
    return {"top_to_bottom": "DOWN", "bottom_to_top": "UP",
            "left_to_right": "RIGHT", "right_to_left": "LEFT"}.get(arrow_in, "IN")


def draw_mode_hud(frame: np.ndarray, ui: UIState, cfg: SceneCfg):
    C_WHITE  = (255, 255, 255)
    C_GRAY   = (160, 158, 155)
    C_BLUE   = (255, 122,   0)
    C_TEAL   = (  0, 200, 220)
    C_ORANGE = (  0, 180, 255)
    C_LIME   = (  0, 220, 100)

    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    PAD   = 18
    LH    = 34
    FS_H  = 0.60
    FS_B  = 0.54
    FS_K  = 0.50
    DOT_R = 6

    if ui.mode == UIMode.NONE:
        badge     = "SHORTCUTS"
        badge_col = C_GRAY
        rows = [
            ("[E]",       "Edit scene",      C_TEAL),
            ("[L]",       "Add lane",        C_LIME),
            ("[I]",       "Change camera",   C_ORANGE),
            ("[S]",       "Save config",     C_BLUE),
            ("[Q]",       "Quit",            C_GRAY),
        ]
        ctx: list = []
    elif ui.mode == UIMode.EDIT_SCENE:
        badge     = "EDIT MODE"
        badge_col = C_TEAL
        rows = [
            ("[R]",       "Draw ROI",        C_ORANGE),
            ("[L]",       "Add lane",        C_LIME),
            ("[D]",       "Delete lane",     C_GRAY),
            ("[C]",       "Clear ROI",       C_GRAY),
            ("[S]",       "Save",            C_BLUE),
            ("[ESC]",     "Exit edit",       C_GRAY),
        ]
        if 0 <= ui.selected_lane < len(cfg.lanes):
            sl  = cfg.lanes[ui.selected_lane]
            ctx = [
                f"Lane: {sl.name}",
                f"Dir:  {_dir_short(sl.arrow_in)}",
                "[Tab]  Cycle direction",
            ]
        elif ui.selected_roi_pt >= 0:
            ctx = [
                f"ROI point {ui.selected_roi_pt} selected",
                "Arrow keys to nudge",
                "Right-click to delete",
            ]
        else:
            ctx = [
                "Drag ROI pts or lane endpoints",
                "Right-click to delete a point",
            ]
    elif ui.mode == UIMode.DRAW_ROI:
        badge     = "DRAW ROI"
        badge_col = C_ORANGE
        rows = [
            ("Click",     "Add ROI point",   C_WHITE),
            ("[R-Click]", "Finish ROI",      C_LIME),
            ("[ESC]",     "Cancel",          C_GRAY),
        ]
        ctx = []
    else:  # ADD_LANE
        badge     = "ADD LANE"
        badge_col = C_LIME
        rows = [
            ("Drag",      "Draw lane line",  C_WHITE),
            ("Release",   "Commit lane",     C_LIME),
            ("[ESC]",     "Cancel",          C_GRAY),
        ]
        ctx = []

    KEY_COL_W  = max(cv2.getTextSize(k, FONT, FS_K, 1)[0][0] for k, _, _ in rows) + 14
    max_desc_w = max(cv2.getTextSize(d, FONT, FS_B, 1)[0][0] for _, d, _ in rows)
    max_ctx_w  = max((cv2.getTextSize(c, FONT, FS_B, 1)[0][0] for c in ctx), default=0)
    (bw, _), _ = cv2.getTextSize(badge, FONT, FS_H, 1)
    badge_w    = DOT_R * 2 + 6 + bw

    W = max(badge_w, KEY_COL_W + max_desc_w, max_ctx_w) + PAD * 2 + 10
    H = (PAD
         + LH
         + 8 + 1 + 10
         + len(rows) * LH
         + (8 + 1 + 10 + len(ctx) * LH if ctx else 0)
         + PAD)

    fh, fw = frame.shape[:2]
    X = fw - W - 10
    Y = 10

    _ios_panel(frame, X, Y, W, H)

    cy = Y + PAD + LH - 6

    cv2.circle(frame, (X + PAD + DOT_R, cy - 4), DOT_R, badge_col, -1, cv2.LINE_AA)
    _put(frame, badge, X + PAD + DOT_R * 2 + 6, cy, FS_H, badge_col)
    cy += 8
    cv2.line(frame, (X + PAD, cy), (X + W - PAD, cy), (60, 58, 55), 1)
    cy += 10

    for key_str, desc, key_col in rows:
        (kw, kh), _ = cv2.getTextSize(key_str, FONT, FS_K, 1)
        kx = X + PAD
        cv2.rectangle(frame, (kx, cy - kh - 3), (kx + kw + 6, cy + 3), (35, 35, 35), cv2.FILLED)
        cv2.rectangle(frame, (kx, cy - kh - 3), (kx + kw + 6, cy + 3), key_col, 1)
        _put(frame, key_str, kx + 3,              cy, FS_K, key_col)
        _put(frame, desc,    X + PAD + KEY_COL_W, cy, FS_B, C_WHITE)
        cy += LH

    if ctx:
        cy += 4
        cv2.line(frame, (X + PAD, cy), (X + W - PAD, cy), (60, 58, 55), 1)
        cy += 10
        for line in ctx:
            col = C_TEAL if line.startswith("[") else C_GRAY
            _put(frame, line, X + PAD, cy, FS_B, col)
            cy += LH


_IP_RE = re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?$')

def _expand_camera_url(s: str, template: str) -> str:
    """Expand a bare IP (e.g. 192.168.1.100) to a full RTSP URL using template.
    Full URLs (rtsp:// or http://) are returned unchanged."""
    s = s.strip()
    if s.startswith("rtsp://") or s.startswith("http"):
        return s
    if _IP_RE.match(s):
        return template.format(host=s)
    return s


def draw_camera_panel(frame: np.ndarray, ui: UIState, current_url: str) -> None:
    """iOS-style panel for live camera URL input."""
    C_WHITE  = (255, 255, 255)
    C_GRAY   = (160, 158, 155)
    C_BLUE   = (255, 122,   0)
    C_TEAL   = (  0, 200, 220)
    C_ORANGE = (  0, 180, 255)
    C_GREEN  = (  0, 220, 100)
    FONT     = cv2.FONT_HERSHEY_SIMPLEX
    PAD      = 18
    LH       = 30

    fh, fw = frame.shape[:2]
    W = min(fw - 40, 620)
    X = (fw - W) // 2
    Y = 20

    C_RED    = (  0,  60, 220)
    n_presets = len(ui.cam_presets)
    error_rows = 1 if ui.cam_input_error else 0
    H = PAD + LH + 14 + LH + 14 + LH + 48 + error_rows * LH + (14 + n_presets * LH if n_presets else 0) + 14 + LH * 2 + PAD

    _ios_panel(frame, X, Y, W, H, alpha=0.88)

    cy = Y + PAD + LH - 6

    # Badge
    cv2.circle(frame, (X + PAD + 6, cy - 4), 6, C_BLUE, -1, cv2.LINE_AA)
    _put(frame, "IP CAMERA", X + PAD + 18, cy, 0.60, C_BLUE)
    cy += 10
    cv2.line(frame, (X + PAD, cy), (X + W - PAD, cy), (60, 58, 55), 1)
    cy += 12

    # Current URL
    _put(frame, "Current:", X + PAD, cy, 0.48, C_GRAY)
    url_display = current_url if len(current_url) <= 55 else "..." + current_url[-52:]
    _put(frame, url_display, X + PAD + 72, cy, 0.48, C_GRAY)
    cy += 14
    cv2.line(frame, (X + PAD, cy), (X + W - PAD, cy), (60, 58, 55), 1)
    cy += 12

    # Input label
    _put(frame, "IP address or full URL:", X + PAD, cy, 0.48, C_WHITE)
    cy += LH

    # Input box — red border on error, teal otherwise
    box_x, box_y = X + PAD, cy - 4
    box_w, box_h = W - PAD * 2, 36
    border_col = C_RED if ui.cam_input_error else C_TEAL
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (45, 42, 40), cv2.FILLED)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), border_col, 1)
    if ui.cam_input_buf:
        buf_display = ui.cam_input_buf if len(ui.cam_input_buf) <= 58 else "..." + ui.cam_input_buf[-55:]
        text_col = C_RED if ui.cam_input_error else C_TEAL
        _put(frame, buf_display + "|", box_x + 8, box_y + 24, 0.48, text_col)
    else:
        _put(frame, "e.g. 192.168.1.100  or  rtsp://...|", box_x + 8, box_y + 24, 0.44, (80, 78, 75))
    cy += 48

    # Error message
    if ui.cam_input_error:
        _put(frame, "Invalid — enter an IP address or rtsp:// / http:// URL", X + PAD, cy, 0.44, C_RED)
        cy += LH

    # Preset cameras
    if n_presets:
        cv2.line(frame, (X + PAD, cy), (X + W - PAD, cy), (60, 58, 55), 1)
        cy += 10
        _put(frame, "Presets — use [↑][↓] to select, [Enter] to connect:", X + PAD, cy, 0.44, C_GRAY)
        cy += LH
        for i, p in enumerate(ui.cam_presets):
            col   = C_GREEN if i == ui.cam_preset_idx else C_GRAY
            arrow = "▶ " if i == ui.cam_preset_idx else "  "
            label = (arrow + p) if len(p) <= 54 else (arrow + "..." + p[-51:])
            _put(frame, label, X + PAD + 10, cy, 0.46, col)
            cy += LH

    # Shortcuts
    cv2.line(frame, (X + PAD, cy), (X + W - PAD, cy), (60, 58, 55), 1)
    cy += 10
    _put(frame, "[Enter]  Connect      [ESC]  Cancel", X + PAD, cy, 0.48, C_ORANGE)
    cy += LH
    _put(frame, "[↑][↓]  Cycle presets           [Backspace]  Delete char", X + PAD, cy, 0.46, C_GRAY)


def _ios_panel(frame: np.ndarray, x: int, y: int, w: int, h: int,
               radius: int = 14, alpha: float = 0.72,
               bg: Tuple[int,int,int] = (18, 15, 12)) -> None:
    """Draw a translucent rounded-corner panel — iOS frosted-glass feel."""
    overlay = frame.copy()
    r = radius
    # Fill: 3 rects + 4 corner circles
    cv2.rectangle(overlay, (x+r, y),   (x+w-r, y+h), bg, cv2.FILLED)
    cv2.rectangle(overlay, (x,   y+r), (x+w,   y+h-r), bg, cv2.FILLED)
    cv2.circle(overlay, (x+r,   y+r),   r, bg, cv2.FILLED)
    cv2.circle(overlay, (x+w-r, y+r),   r, bg, cv2.FILLED)
    cv2.circle(overlay, (x+r,   y+h-r), r, bg, cv2.FILLED)
    cv2.circle(overlay, (x+w-r, y+h-r), r, bg, cv2.FILLED)
    # Blend panel area only
    roi = np.s_[y:y+h, x:x+w]
    frame[roi] = cv2.addWeighted(overlay[roi], alpha, frame[roi], 1-alpha, 0)
    # Crisp thin border drawn after blend
    border = (70, 65, 62)
    cv2.rectangle(frame, (x+r, y),   (x+w-r, y),   border, 1)
    cv2.rectangle(frame, (x+r, y+h), (x+w-r, y+h), border, 1)
    cv2.rectangle(frame, (x,   y+r), (x,   y+h-r), border, 1)
    cv2.rectangle(frame, (x+w, y+r), (x+w, y+h-r), border, 1)


def _put(frame, text, x, y, scale, color, thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_count_panel(frame: np.ndarray,
                     counts: Dict[str, Tuple[int,int]],
                     fps: float,
                     lanes: List[LaneCfg],
                     day_mode: bool = True):
    # ── iOS colour tokens (BGR) ──────────────────────
    C_WHITE   = (255, 255, 255)
    C_GRAY    = (160, 158, 155)
    C_BLUE    = (255, 122,   0)   # iOS blue
    C_GREEN   = ( 89, 199,  52)   # iOS green
    C_RED     = ( 48,  59, 255)   # iOS red
    C_YELLOW  = (  0, 204, 255)   # iOS yellow
    C_DAY     = (  0, 220, 255)   # yellow (day)
    C_NIGHT   = (120,  60,  20)   # dark blue (night)

    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    PAD   = 18
    LH    = 36    # line height
    FS_H  = 0.64  # header font scale
    FS_B  = 0.58  # body font scale

    # Count only rows that have activity
    active = [(n, ic, oc) for n, (ic, oc) in sorted(counts.items()) if ic+oc > 0]
    n_rows = len(active) if active else 1

    # Panel geometry
    W = 340
    H = (PAD                  # top padding
         + LH                 # title row  (FPS + mode)
         + 6                  # gap
         + 1                  # divider
         + 6                  # gap
         + LH                 # column header
         + n_rows * LH        # class rows
         + (14 + LH * len([l for l in lanes if l.enabled]) if lanes else 0)
         + PAD)               # bottom padding
    X, Y = 12, 12

    _ios_panel(frame, X, Y, W, H)

    cy = Y + PAD + 14   # cursor y (baseline)

    # ── Header row: FPS  ·  model indicator ─────────
    _put(frame, f"{fps:.1f} fps", X+PAD, cy, FS_H, C_BLUE)
    dot_col   = C_DAY if day_mode else C_NIGHT
    mode_str  = "Day" if day_mode else "Night"
    (mw, _), _ = cv2.getTextSize(mode_str, FONT, FS_H, 1)
    dot_cx    = X + W - PAD - mw - 12
    cv2.circle(frame, (dot_cx, cy-5), 5, dot_col, -1, cv2.LINE_AA)
    _put(frame, mode_str, dot_cx+9, cy, FS_H, C_GRAY)

    cy += 6
    # divider
    cv2.line(frame, (X+PAD, cy), (X+W-PAD, cy), (60, 58, 55), 1)
    cy += 10

    # ── Column headers ───────────────────────────────
    _put(frame, "Class",  X+PAD,       cy, 0.44, C_GRAY)
    _put(frame, "In",     X+W-PAD-72,  cy, 0.44, C_GRAY)
    _put(frame, "Out",    X+W-PAD-26,  cy, 0.44, C_GRAY)
    cy += LH

    # ── Class rows ───────────────────────────────────
    if not active:
        _put(frame, "No detections yet", X+PAD, cy, FS_B, C_GRAY)
        cy += LH
    else:
        for name, in_cnt, out_cnt in active:
            _put(frame, name, X+PAD, cy, FS_B, C_WHITE)

            # IN count (green)
            in_str = str(in_cnt)
            (iw, _), _ = cv2.getTextSize(in_str, FONT, FS_B, 1)
            _put(frame, in_str, X+W-PAD-72-iw, cy, FS_B,
                 C_GREEN if in_cnt > 0 else C_GRAY)

            # OUT count (red)
            out_str = str(out_cnt)
            (ow, _), _ = cv2.getTextSize(out_str, FONT, FS_B, 1)
            _put(frame, out_str, X+W-PAD-26-ow, cy, FS_B,
                 C_RED if out_cnt > 0 else C_GRAY)

            cy += LH

    # ── Lane totals ──────────────────────────────────
    active_lanes = [l for l in lanes if l.enabled]
    if active_lanes:
        cy += 2
        cv2.line(frame, (X+PAD, cy), (X+W-PAD, cy), (60, 58, 55), 1)
        cy += 10
        for lane in active_lanes:
            _put(frame, lane.name, X+PAD, cy, FS_B, C_GRAY)
            in_s  = str(lane.cross_in)
            out_s = str(lane.cross_out)
            (iw2, _), _ = cv2.getTextSize(in_s,  FONT, FS_B, 1)
            (ow2, _), _ = cv2.getTextSize(out_s, FONT, FS_B, 1)
            _put(frame, in_s,  X+W-PAD-72-iw2, cy, FS_B, C_GREEN)
            _put(frame, out_s, X+W-PAD-26-ow2, cy, FS_B, C_RED)
            cy += LH


def draw_label(frame: np.ndarray, rect, text: str, color):
    x, y, w, h = rect
    (lw, lh), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    label_y = max(y, lh + 4)
    cv2.rectangle(frame, (x, label_y-lh-4), (x+lw+4, label_y+bl), color, cv2.FILLED)
    cv2.putText(frame, text, (x+2, label_y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,0), 1, cv2.LINE_AA)


# ============================================================
#  CSV Logger
# ============================================================
class CsvLogger:
    def __init__(self, path: str):
        self.path = path
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._f  = open(path, 'a', newline='')
        self._w  = csv.writer(self._f)
        if is_new:
            self._w.writerow(["timestamp", "track_id", "class", "direction", "lane"])
            self._f.flush()

    def log(self, track_id: int, cls: str, direction: str, lane_name: str = ""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._w.writerow([ts, track_id, cls, direction, lane_name])
        self._f.flush()

    def close(self):
        self._f.close()


# ============================================================
#  Live Stats Writer  (feeds live_stats.py dashboard)
# ============================================================
def _derive_model_name(onnx_file: str, cfg_file: str) -> str:
    if onnx_file:
        import re as _re
        p = onnx_file.replace("\\", "/").lower()
        for arch in ("yolov8n", "yolov8s", "yolo11n", "yolo11s"):
            if arch in p:
                run_m = _re.search(r'/(run\d+)/', p)
                run   = run_m.group(1) if run_m else ""
                return f"{arch} {run}".strip()
    if cfg_file:
        base = os.path.basename(cfg_file).lower()
        if "v4-tiny" in base or "v4tiny" in base:
            return "YOLOv4-tiny"
        if "v4" in base:
            return "YOLOv4"
    return "Unknown"


def _load_model_bg(preset: dict, conf: float, nms: float, result: list,
                   use_gpu: bool = True) -> None:
    """Background thread: load a model preset into result[0]/result[1]."""
    new_det = YoloDetector()
    new_det.conf_thresh = conf
    new_det.nms_thresh  = nms
    new_det.net_size    = int(preset.get('input_size', 416))
    has_onnx = bool(preset.get('onnx') and os.path.exists(preset['onnx']))
    ok = False
    try:
        if has_onnx:
            ok = new_det.load(names_file=preset['names'],
                              onnx_file=preset['onnx'], use_gpu=use_gpu)
        elif preset.get('cfg') and preset.get('weights'):
            ok = new_det.load(preset['cfg'], preset['weights'],
                              preset['names'], use_gpu=use_gpu)
    except Exception as e:
        print(f"[SWITCH] Load error: {e}")
    result[0] = new_det if ok else None
    result[1] = preset.get('label', 'Unknown')
    if ok:
        print(f"[SWITCH] {result[1]} ready — swapping next frame")
    else:
        print(f"[SWITCH] Failed to load {result[1]}")


class StatsWriter:
    WRITE_INTERVAL = 1.0   # seconds between JSON writes

    def __init__(self, path: str, model_name: str, source: str, conf: float):
        self.path        = path
        self.model_name  = model_name
        self.source      = source
        self.conf        = conf
        self._start      = time.time()
        self._last_write = 0.0
        self._events: list = []    # last 20 crossing events

    def add_event(self, cls_name: str, direction: str, lane_name: str, conf: float):
        ts = datetime.now().strftime("%H:%M:%S")
        self._events.append({
            "ts":   ts,
            "cls":  cls_name,
            "dir":  direction,
            "lane": lane_name,
            "conf": round(conf, 2),
        })
        if len(self._events) > 20:
            self._events.pop(0)

    def tick(self, fps: float, frame_idx: int,
             tracks: list, class_counts: dict, scene_cfg,
             frame_dets: dict = None):
        now = time.time()
        if now - self._last_write < self.WRITE_INTERVAL:
            return
        self._last_write = now

        active = sum(1 for t in tracks if t.hit_streak >= MIN_HIT_STREAK)
        confs  = [t.confidence for t in tracks if t.hit_streak >= MIN_HIT_STREAK]
        avg_conf = round(sum(confs) / len(confs), 2) if confs else 0.0

        lanes_data = [
            {"name": l.name, "in": l.cross_in, "out": l.cross_out}
            for l in scene_cfg.lanes if l.enabled
        ]

        data = {
            "ts":            now,
            "fps":           round(fps, 1),
            "frame":         frame_idx,
            "uptime_s":      int(now - self._start),
            "model":         self.model_name,
            "source":        self.source,
            "conf_thresh":   self.conf,
            "active_tracks": active,
            "avg_conf":      avg_conf,
            "frame_dets":    frame_dets or {},      # what model sees THIS frame
            "class_counts":  {k: list(v) for k, v in class_counts.items()},
            "lanes":         lanes_data,
            "events":        list(reversed(self._events))[:15],
        }
        try:
            tmp = self.path + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except Exception:
            pass


# ============================================================
#  ROI mask helper
# ============================================================
def in_roi(pt: Tuple[float,float], roi_pts) -> bool:
    if not roi_pts:
        return True
    pts = np.array([[p[0], p[1]] for p in roi_pts], dtype=np.float32)
    return cv2.pointPolygonTest(pts, (float(pt[0]), float(pt[1])), False) >= 0


# ============================================================
#  main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Vehicle Counter — cross-platform")
    p.add_argument("input",          help="Video file path, RTSP URL, or webcam index (0)")
    p.add_argument("--cfg",          default="", help="YOLOv4 .cfg file")
    p.add_argument("--weights",      default="", help="YOLOv4 .weights file")
    p.add_argument("--names",        default="", help="Class names file")
    p.add_argument("--onnx",         default="", help="YOLOv8 ONNX model file (replaces --cfg/--weights)")
    p.add_argument("--config",       default="", help="scene_config.json path")
    p.add_argument("--size",   type=int,   default=416, help="YOLO input size (default 416)")
    p.add_argument("--conf",   type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--nms",    type=float, default=0.40, help="NMS threshold")
    p.add_argument("--csv",          default="vehicle_counts.csv", help="CSV output path")
    p.add_argument("--out",          default="", help="Save annotated video to file")
    p.add_argument("--nowin",  action="store_true", help="Headless mode")
    p.add_argument("--cpu",    action="store_true", help="Force CPU backend")
    p.add_argument("--gpu",    action="store_true", help="Force GPU (CUDA) backend")
    p.add_argument("--stats",  default="live_stats.json",
                   help="Write live stats JSON for dashboard (default: live_stats.json)")
    p.add_argument("--skip",   type=int, default=1,
                   help="Process 1 out of every N frames (default 1 = no skip). Use 2-3 to reduce CPU load on slow hardware.")
    return p.parse_args()


def is_day_time() -> bool:
    return 6 <= datetime.now().hour < 18


def make_night_path(p: str) -> str:
    return p.replace("-day.", "-night.")


def _open_rtsp_cap(url: str, timeout_ms: int = 5000) -> cv2.VideoCapture:
    """Open an RTSP/HTTP stream with a short connection timeout (default 5 s)."""
    cap = cv2.VideoCapture()
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
    cap.open(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.open(url)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    return cap


def main():
    args = parse_args()

    # --- Resolve config paths ---
    cfg_file     = args.cfg
    weights_file = args.weights
    names_file   = args.names
    onnx_file    = args.onnx
    scene_path   = args.config

    # If scene_config.json provided, read YOLO paths from it when not overridden
    if scene_path and os.path.exists(scene_path):
        scene_dir    = os.path.dirname(os.path.abspath(scene_path))
        scene_parent = os.path.dirname(scene_dir)  # one level up (counting_app/)

        def resolve_model_path(raw: str) -> str:
            """Try path as-is, then relative to scene dir, then relative to parent dir."""
            if os.path.isabs(raw):
                return raw
            for base in (scene_dir, scene_parent, os.getcwd()):
                candidate = os.path.join(base, raw)
                if os.path.exists(candidate):
                    return candidate
            # Return best guess (scene_dir) even if not found yet
            return os.path.join(scene_parent, raw)

        with open(scene_path) as f:
            _sc = json.load(f)
        yolo_sec = _sc.get('yolo', {})
        if not onnx_file and yolo_sec.get('onnx', ''):
            onnx_file = resolve_model_path(yolo_sec.get('onnx', ''))
        if not cfg_file:
            cfg_file = resolve_model_path(yolo_sec.get('cfg', ''))
        if not weights_file:
            weights_file = resolve_model_path(yolo_sec.get('weights', ''))
        if not names_file:
            names_file = resolve_model_path(yolo_sec.get('names', ''))
        if not args.conf or args.conf == 0.35:
            args.conf = float(yolo_sec.get('conf', args.conf))
        if not args.nms or args.nms == 0.40:
            args.nms  = float(yolo_sec.get('nms',  args.nms))
        if args.size == 416:
            args.size = int(yolo_sec.get('input_size', args.size))

        # Load model presets for live switching (keys 1/2/3)
        model_presets = _sc.get('model_presets', [])
    else:
        model_presets = []

    using_onnx = bool(onnx_file and os.path.exists(onnx_file))
    if not using_onnx and (not cfg_file or not weights_file or not names_file):
        print("[ERROR] Provide --onnx <file> or --cfg/--weights/--names, or point --config to a scene_config.json.", file=sys.stderr)
        sys.exit(1)

    # Night model (darknet only — ONNX uses a single model file)
    has_night    = False
    active_cfg     = cfg_file
    active_weights = weights_file
    day_mode       = is_day_time()   # always reflects real clock, not model availability
    if not using_onnx:
        night_cfg     = make_night_path(cfg_file)
        night_weights = make_night_path(weights_file)
        has_night     = os.path.exists(night_cfg) and os.path.exists(night_weights)
        if has_night:
            print(f"[INFO] Night model found: {night_cfg}")
        active_cfg     = cfg_file     if day_mode else night_cfg
        active_weights = weights_file if day_mode else night_weights

    print(f"[INFO] Loading {'YOLOv8 ONNX' if using_onnx else ('DAY' if day_mode else 'NIGHT') + ' (darknet)'} model.")

    # --- Load detector ---
    detector = YoloDetector()
    detector.conf_thresh = args.conf
    detector.nms_thresh  = args.nms
    detector.net_size    = args.size
    if using_onnx:
        if not detector.load(names_file=names_file, onnx_file=onnx_file,
                             use_gpu=not args.cpu, force_gpu=args.gpu):
            sys.exit(1)
    else:
        if not detector.load(active_cfg, active_weights, names_file,
                             use_gpu=not args.cpu, force_gpu=args.gpu):
            sys.exit(1)

    # --- Load scene config ---
    scene_cfg = SceneCfg()
    if scene_path and os.path.exists(scene_path):
        scene_cfg = SceneCfg.load(scene_path)

    # Initial class counts dict (ordered by class name list, skipping filtered classes)
    class_counts: Dict[str, Tuple[int,int]] = {
        name: (0, 0) for name in detector.class_names if name not in SKIP_CLASSES
    }

    # --- Open video ---
    input_str = args.input
    is_stream = input_str.startswith("rtsp://") or input_str.startswith("http")

    if input_str.isdigit():
        cap = cv2.VideoCapture(int(input_str))
    elif is_stream:
        cap = _open_rtsp_cap(input_str)
    else:
        cap = cv2.VideoCapture(input_str)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_str}", file=sys.stderr)
        sys.exit(1)

    src_fps  = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0 or src_fps > 200:
        src_fps = 25.0
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {frame_w}x{frame_h}  FPS={src_fps:.1f}  frames={total_fr}")

    # Warm up stream
    if is_stream:
        for _ in range(5):
            cap.read()

    # --- Video writer ---
    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.out, fourcc, src_fps, (frame_w, frame_h))

    # --- CSV logger ---
    csv_logger = CsvLogger(args.csv)

    # --- Live stats writer ---
    _model_label = _derive_model_name(onnx_file, cfg_file)
    stats_writer = StatsWriter(args.stats, _model_label, input_str, args.conf)

    # --- Tracker ---
    tracker = CentroidTracker()

    # --- UI state ---
    ui = UIState()
    # Load camera presets from scene_config.json cameras list
    if scene_path and os.path.exists(scene_path):
        try:
            with open(scene_path) as _f:
                _sc_raw = json.load(_f)
            if _sc_raw.get('rtsp_template'):
                ui.cam_rtsp_template = _sc_raw['rtsp_template']
            for _c in _sc_raw.get('cameras', []):
                if isinstance(_c, str):
                    ui.cam_presets.append(_c)
                elif isinstance(_c, dict):
                    _url = _c.get('rtsp') or _c.get('url') or ''
                    if _url:
                        ui.cam_presets.append(_url)
        except Exception:
            pass
    if is_stream and input_str not in ui.cam_presets:
        ui.cam_presets.insert(0, input_str)
    WIN = "Vehicle Counter"

    if not args.nowin:
        # Windows: set DPI awareness BEFORE creating any window.
        # Without this, Windows auto-scales the app, making mouse coordinates
        # inconsistent with where things are drawn on screen.
        if platform.system() == "Windows":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
            except Exception:
                try:
                    ctypes.windll.user32.SetProcessDPIAware()   # fallback (Win7+)
                except Exception:
                    pass

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

        # Size the window to fit the screen on first open, and set the initial
        # scale so mouse callbacks work correctly before the first imshow.
        _win_w, _win_h = frame_w, frame_h
        if platform.system() == "Windows":
            try:
                import ctypes
                _sw = ctypes.windll.user32.GetSystemMetrics(0)   # screen width
                _sh = ctypes.windll.user32.GetSystemMetrics(1)   # screen height
                _avail_h = _sh - 80    # leave room for taskbar
                if _win_w > _sw - 20 or _win_h > _avail_h:
                    _s = min((_sw - 20) / _win_w, _avail_h / _win_h)
                    _win_w, _win_h = int(_win_w * _s), int(_win_h * _s)
            except Exception:
                _win_w, _win_h = min(frame_w, 1280), min(frame_h, 720)
        elif frame_w > 1920 or frame_h > 1080:
            _s = min(1920 / frame_w, 1080 / frame_h)
            _win_w, _win_h = int(frame_w * _s), int(frame_h * _s)

        cv2.resizeWindow(WIN, _win_w, _win_h)
        # Set initial coordinate scale so clicks work before the first frame arrives
        ui.scale_x = frame_w / _win_w
        ui.scale_y = frame_h / _win_h

        cb = make_mouse_callback(ui, scene_cfg, (frame_w, frame_h))
        cv2.setMouseCallback(WIN, cb)

    print("=" * 44)
    print(" Vehicle Counter  (OpenCV DNN)")
    print("=" * 44)
    print(f" Input  : {input_str}")
    _backend = "CPU (forced)" if args.cpu else "GPU (CUDA)" if args.gpu else "GPU if available, CPU fallback"
    print(f" Backend: {_backend}")
    print(f" CSV    : {args.csv}")
    if scene_path:
        print(f" SceneCfg: {scene_path}  ({len(scene_cfg.lanes)} lanes)")
    print("=" * 44)
    if not args.nowin:
        print("Keys: [E] Edit  [L] Add Lane  [S] Save  [Q/ESC] Quit")

    # Background model-switch state: [new_detector | None, label]
    _sw_result: list = [None, None]
    _sw_thread: threading.Thread | None = None

    frame_idx        = 0
    fps_display      = 0.0
    last_model_check = time.time()
    raw_key          = -1
    _reconnect_n     = 0        # consecutive failed reads; reset on success
    _cam_failed      = False    # True after giving up — stops retrying until user changes URL
    last_good_frame: Optional[np.ndarray] = None
    _scale_synced    = False    # True once Win32 client rect confirmed on first frame

    while True:
        t0 = time.perf_counter()

        # --- Key handling (top of loop so it runs even during stream reconnects) ---
        if not args.nowin:
            raw_key = cv2.waitKey(1)
            key     = raw_key & 0xFF
        else:
            raw_key = -1
            key     = 0xFF

        if (key in (ord('q'), ord('Q')) or (key == 27 and ui.mode == UIMode.NONE)) and ui.mode != UIMode.CAMERA_INPUT:
            break
        elif key == 27:  # ESC cancels sub-mode
            if ui.mode in (UIMode.DRAW_ROI, UIMode.ADD_LANE):
                ui.roi_draft = []
                ui.add_lane_dragging = False
            ui.mode = UIMode.NONE
        elif ui.mode == UIMode.CAMERA_INPUT:
            if key == 13:  # Enter — connect to IP camera
                raw_input = (ui.cam_presets[ui.cam_preset_idx]
                             if 0 <= ui.cam_preset_idx < len(ui.cam_presets)
                             else ui.cam_input_buf.strip())
                new_url = _expand_camera_url(raw_input, ui.cam_rtsp_template) if raw_input else ""
                if new_url:
                    if not (new_url.startswith("rtsp://") or new_url.startswith("http")):
                        ui.cam_input_error = True
                    else:
                        print(f"[CAM] Reconnecting to: {new_url}")
                        cap.release()
                        input_str = new_url
                        is_stream = True
                        _reconnect_n = 0
                        _cam_failed = False
                        _scale_synced = False
                        cap = _open_rtsp_cap(input_str)
                        ui.mode = UIMode.NONE
            elif key in (8, 127):  # Backspace / Delete
                ui.cam_input_buf = ui.cam_input_buf[:-1]
                ui.cam_input_error = False
                ui.cam_preset_idx = -1
            elif raw_key in (82, 2490368):  # Up arrow
                if ui.cam_presets:
                    ui.cam_preset_idx = (ui.cam_preset_idx - 1) % len(ui.cam_presets)
                    ui.cam_input_buf = ui.cam_presets[ui.cam_preset_idx]
                    ui.cam_input_error = False
            elif raw_key in (84, 2621440):  # Down arrow
                if ui.cam_presets:
                    ui.cam_preset_idx = (ui.cam_preset_idx + 1) % len(ui.cam_presets)
                    ui.cam_input_buf = ui.cam_presets[ui.cam_preset_idx]
                    ui.cam_input_error = False
            elif 32 <= key < 127:  # Printable ASCII — build URL
                ui.cam_input_buf += chr(key)
                ui.cam_input_error = False
                ui.cam_preset_idx = -1
        elif key == ord('i') or key == ord('I'):
            ui.cam_input_buf = ""
            ui.cam_input_error = False
            ui.cam_preset_idx = -1
            ui.mode = UIMode.CAMERA_INPUT
        elif key == ord('e') or key == ord('E'):
            ui.mode = UIMode.EDIT_SCENE if ui.mode != UIMode.EDIT_SCENE else UIMode.NONE
        elif key == ord('l') or key == ord('L'):
            ui.mode = UIMode.ADD_LANE
        elif key == ord('r') or key == ord('R'):
            if ui.mode == UIMode.EDIT_SCENE:
                ui.roi_draft = list(scene_cfg.roi_pts)
                scene_cfg.roi_pts = []
                scene_cfg.roi_enabled = False
                ui.mode = UIMode.DRAW_ROI
        elif key == ord('c') or key == ord('C'):
            if ui.mode == UIMode.EDIT_SCENE:
                scene_cfg.roi_pts = []
                scene_cfg.roi_enabled = False
                ui.need_save = True
        elif key == ord('d') or key == ord('D'):
            if ui.mode == UIMode.EDIT_SCENE and 0 <= ui.selected_lane < len(scene_cfg.lanes):
                scene_cfg.lanes.pop(ui.selected_lane)
                ui.selected_lane = -1
                ui.need_save = True
        elif key == ord('s') or key == ord('S'):
            if scene_path:
                scene_cfg.save()
        elif key == 9:  # Tab – cycle arrow_in direction
            if ui.mode == UIMode.EDIT_SCENE and 0 <= ui.selected_lane < len(scene_cfg.lanes):
                dirs = ["top_to_bottom", "bottom_to_top", "left_to_right", "right_to_left"]
                lane = scene_cfg.lanes[ui.selected_lane]
                cur  = dirs.index(lane.arrow_in) if lane.arrow_in in dirs else 0
                lane.arrow_in = dirs[(cur + 1) % len(dirs)]
                ui.need_save = True
        elif key in (ord('1'), ord('2'), ord('3')):
            preset_idx = key - ord('1')
            if preset_idx < len(model_presets):
                p = model_presets[preset_idx]
                print(f"[SWITCH] Loading preset {preset_idx+1}: {p.get('label','?')}")
                has_onnx = bool(p.get('onnx') and os.path.exists(p['onnx']))
                ok = False
                if has_onnx:
                    ok = detector.load(names_file=p['names'], onnx_file=p['onnx'], use_gpu=False)
                elif p.get('cfg') and p.get('weights'):
                    ok = detector.load(p['cfg'], p['weights'], p['names'], use_gpu=False)
                if ok:
                    detector.net_size       = int(p.get('input_size', 416))
                    stats_writer.model_name = p.get('label', 'Unknown')
                    print(f"[SWITCH] Now using: {stats_writer.model_name}")
                else:
                    print(f"[SWITCH] Failed — check model path in scene_config.json")
            else:
                print(f"[SWITCH] Preset {preset_idx+1} not defined in scene_config.json")

        # Arrow keys: nudge selected ROI point or lane (3px per press)
        _ARROW = {
            81: (-3, 0), 83: (3, 0), 82: (0, -3), 84: (0, 3),
            2424832: (-3, 0), 2555904: (3, 0), 2490368: (0, -3), 2621440: (0, 3),
        }
        _adxdy = _ARROW.get(raw_key) or _ARROW.get(key)
        if _adxdy and ui.mode == UIMode.EDIT_SCENE:
            dx, dy = _adxdy
            if 0 <= ui.selected_roi_pt < len(scene_cfg.roi_pts):
                p = scene_cfg.roi_pts[ui.selected_roi_pt]
                scene_cfg.roi_pts[ui.selected_roi_pt] = _clamp_pt((p[0]+dx, p[1]+dy), frame_w, frame_h)
                ui.need_save = True
            elif 0 <= ui.selected_lane < len(scene_cfg.lanes):
                ln = scene_cfg.lanes[ui.selected_lane]
                ln.x1, ln.y1 = _clamp_pt((ln.x1+dx, ln.y1+dy), frame_w, frame_h)
                ln.x2, ln.y2 = _clamp_pt((ln.x2+dx, ln.y2+dy), frame_w, frame_h)
                ui.need_save = True

        if _cam_failed:
            # Stream permanently failed — show last good frame and wait for user to change URL
            if last_good_frame is not None:
                frame = last_good_frame.copy()
            else:
                frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        else:
            ret, frame = cap.read()
            if not ret:
                if is_stream:
                    _reconnect_n += 1
                    _RECONNECT_LIMIT = 5
                    if _reconnect_n > _RECONNECT_LIMIT:
                        print(f"[ERROR] Cannot reach {input_str} — check IP / network")
                        _cam_failed = True
                        _reconnect_n = 0
                        if not args.nowin:
                            ui.cam_input_buf = ""
                            ui.cam_input_error = False
                            ui.cam_preset_idx = -1
                            ui.mode = UIMode.CAMERA_INPUT
                    else:
                        delay = min(0.5 * (2 ** (_reconnect_n - 1)), 8.0)
                        print(f"[WARN] Stream lost (attempt {_reconnect_n}/{_RECONNECT_LIMIT}), retry in {delay:.1f}s...")
                        time.sleep(delay)
                        cap.release()
                        cap = _open_rtsp_cap(input_str)
                    continue
                break  # end of file
            _reconnect_n = 0
            last_good_frame = frame

            # Drain buffered frames on live streams to stay at the live edge.
            # Grab (decode-skip) all queued frames; keep only the last decoded one.
            if is_stream:
                drained = 0
                while cap.grab():
                    drained += 1
                    if drained >= 8:
                        break
                if drained > 0:
                    ret2, frame2 = cap.retrieve()
                    if ret2 and frame2 is not None:
                        frame = frame2
                        last_good_frame = frame

        frame_idx += 1

        # Frame-skip: process every Nth frame only (reduces CPU load on slow hardware).
        if args.skip > 1 and frame_idx % args.skip != 0:
            if not args.nowin:
                cv2.imshow(WIN, frame if last_good_frame is None else last_good_frame)
            continue

        # Day/Night check (every 60s) — updates panel indicator always; also swaps darknet model if available
        if (time.time() - last_model_check) > 60:
            last_model_check = time.time()
            new_day = is_day_time()
            if new_day != day_mode:
                day_mode = new_day
                if has_night:
                    active_cfg     = cfg_file     if day_mode else night_cfg
                    active_weights = weights_file if day_mode else night_weights
                    print(f"[INFO] Switching to {'DAY' if day_mode else 'NIGHT'} model.")
                    detector.load(active_cfg, active_weights, names_file, use_gpu=not args.cpu)

        # --- Run detection (skip ROI mask for detection; filter in tracker update) ---
        dets = detector.detect(frame)

        # Filter detections by ROI
        if scene_cfg.roi_enabled and scene_cfg.roi_pts:
            dets = [d for d in dets if in_roi(d.centroid(), scene_cfg.roi_pts)]

        # Build current-frame detection counts for dashboard
        frame_dets: Dict[str, int] = {}
        for d in dets:
            cn = detector.class_name(d.class_id)
            if cn in SKIP_CLASSES:
                continue
            frame_dets[cn] = frame_dets.get(cn, 0) + 1

        # --- Update tracker ---
        tracks = tracker.update(dets)

        # --- Lane crossing ---
        for trk in tracks:
            if trk.hit_streak < MIN_HIT_STREAK:
                continue
            for lane in scene_cfg.lanes:
                if not lane.enabled or lane.id in trk.counted_lanes:
                    continue
                result = lane_crossed(trk.prev_center, trk.center, lane)
                if result == 0:
                    continue
                trk.counted_lanes.add(lane.id)
                cls_name = detector.class_name(trk.class_id)
                if cls_name in SKIP_CLASSES:
                    continue
                if result == +1 and lane.count_in:
                    lane.cross_in += 1
                    lane.flash_frames = 8
                    in_cnt, out_cnt = class_counts.get(cls_name, (0,0))
                    class_counts[cls_name] = (in_cnt+1, out_cnt)
                    csv_logger.log(trk.id, cls_name, "in", lane.name)
                    stats_writer.add_event(cls_name, "IN", lane.name, trk.confidence)
                    print(f"[COUNT] {cls_name} IN  lane={lane.name}  track={trk.id}")
                elif result == -1 and lane.count_out:
                    lane.cross_out += 1
                    lane.flash_frames = 8
                    in_cnt, out_cnt = class_counts.get(cls_name, (0,0))
                    class_counts[cls_name] = (in_cnt, out_cnt+1)
                    csv_logger.log(trk.id, cls_name, "out", lane.name)
                    stats_writer.add_event(cls_name, "OUT", lane.name, trk.confidence)
                    print(f"[COUNT] {cls_name} OUT lane={lane.name}  track={trk.id}")

        # Decrement flash counters
        for lane in scene_cfg.lanes:
            if lane.flash_frames > 0:
                lane.flash_frames -= 1

        # --- Draw ---
        if not args.nowin or writer:
            # Bounding boxes
            for trk in tracks:
                if trk.hit_streak < MIN_HIT_STREAK:
                    continue
                col  = class_color(trk.class_id)
                x, y, w, h = trk.rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), col, 2)
                label = f"{detector.class_name(trk.class_id)} {trk.confidence:.2f}"
                draw_label(frame, trk.rect, label, col)
                cx, cy = int(trk.center[0]), int(trk.center[1])
                cv2.circle(frame, (cx, cy), 4, col, -1, cv2.LINE_AA)

            draw_scene_overlay(frame, scene_cfg, ui)
            draw_count_panel(frame, class_counts, fps_display, scene_cfg.lanes, day_mode)
            if not args.nowin:
                if ui.mode == UIMode.CAMERA_INPUT:
                    draw_camera_panel(frame, ui, input_str)
                else:
                    draw_mode_hud(frame, ui, scene_cfg)

            if writer:
                writer.write(frame)

            if not args.nowin:
                cv2.imshow(WIN, frame)
                # On the very first rendered frame, force a Win32 scale re-sync so
                # the initial estimate (from GetSystemMetrics) is replaced with the
                # true client rect, which accounts for window decorations and DPI.
                if platform.system() == "Windows" and not _scale_synced:
                    cv2.waitKey(30)   # let the window actually paint
                    _wcs0 = _win32_client_size(WIN)
                    if _wcs0:
                        sx0 = frame_w / _wcs0[0]
                        sy0 = frame_h / _wcs0[1]
                        if 0.05 < sx0 < 20 and 0.05 < sy0 < 20:
                            ui.scale_x = sx0
                            ui.scale_y = sy0
                    _scale_synced = True

                # Keep scale in sync when the user resizes the window.
                # On Windows use Win32 GetClientRect (physical pixels, same space
                # as WM_MOUSEMOVE). On macOS/Linux use getWindowImageRect.
                if platform.system() == "Windows":
                    _wcs = _win32_client_size(WIN)
                    if _wcs:
                        sx = frame_w / _wcs[0]
                        sy = frame_h / _wcs[1]
                        if 0.05 < sx < 20 and 0.05 < sy < 20:
                            ui.scale_x = sx
                            ui.scale_y = sy
                else:
                    try:
                        wr = cv2.getWindowImageRect(WIN)
                        if wr[2] > 10 and wr[3] > 10:
                            sx = frame_w / wr[2]
                            sy = frame_h / wr[3]
                            if 0.05 < sx < 20 and 0.05 < sy < 20:
                                ui.scale_x = sx
                                ui.scale_y = sy
                    except Exception:
                        pass

        # --- Auto-save ---
        if ui.need_save and scene_path:
            scene_cfg.save()
            ui.need_save = False

        # --- FPS ---
        elapsed = time.perf_counter() - t0
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(elapsed, 1e-6))

        # --- Live stats ---
        stats_writer.tick(fps_display, frame_idx, tracks, class_counts, scene_cfg, frame_dets)

        # --- Swap pending model if background load finished ---
        if _sw_result[0] is not None and (_sw_thread is None or not _sw_thread.is_alive()):
            detector                = _sw_result[0]
            stats_writer.model_name = _sw_result[1]
            _sw_result[0]           = None
            print(f"[SWITCH] Now using: {stats_writer.model_name}")

        # --- Command file (model switch from terminal) ---
        _cmd_path = os.path.join(os.path.dirname(os.path.abspath(args.stats)), "model_cmd.txt")
        if os.path.exists(_cmd_path) and (_sw_thread is None or not _sw_thread.is_alive()):
            try:
                with open(_cmd_path) as _f:
                    _cmd = _f.read().strip()
                os.remove(_cmd_path)
                # Re-read presets fresh so no restart needed
                if scene_path and os.path.exists(scene_path):
                    with open(scene_path) as _f2:
                        model_presets = json.load(_f2).get('model_presets', [])
                _pidx = int(_cmd) - 1
                if 0 <= _pidx < len(model_presets):
                    _p = model_presets[_pidx]
                    print(f"[SWITCH] Loading {_p.get('label','?')} in background...")
                    _sw_result = [None, None]
                    _sw_thread = threading.Thread(
                        target=_load_model_bg,
                        args=(_p, detector.conf_thresh, detector.nms_thresh, _sw_result, not args.cpu),
                        daemon=True,
                    )
                    _sw_thread.start()
                else:
                    print(f"[SWITCH] Preset {_cmd} not found ({len(model_presets)} presets loaded)")
            except Exception as _e:
                print(f"[SWITCH] Error: {_e}")


    # Cleanup
    cap.release()
    if writer:
        writer.release()
    csv_logger.close()
    if not args.nowin:
        cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
