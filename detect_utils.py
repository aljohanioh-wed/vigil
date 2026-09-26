import os
import time
from collections import deque, defaultdict
from datetime import datetime

import cv2
import numpy as np

FIRE_LOWER = np.array([0, 140, 170], dtype=np.uint8)
FIRE_UPPER = np.array([28, 255, 255], dtype=np.uint8)
MIN_FIRE_AREA = 1200
FIRE_FILL_RATIO = 0.30
FIRE_ASPECT_MIN = 0.30
FIRE_ASPECT_MAX = 3.2

SMOKE_LOWER = np.array([0, 0, 90], dtype=np.uint8)
SMOKE_UPPER = np.array([180, 40, 200], dtype=np.uint8)
MIN_SMOKE_AREA = 2500

FACE_MIN_SIZE = (40, 40)
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 6


def _clip_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(x1, width - 1)); y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1)); y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def detect_fire_regions(frame_bgr):
    height, width = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, FIRE_LOWER, FIRE_UPPER)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_FIRE_AREA: continue
        x, y, w, h = cv2.boundingRect(c)
        fill_ratio = area / float(w * h + 1e-6)
        if fill_ratio < FIRE_FILL_RATIO: continue
        aspect = w / float(h + 1e-6)
        if aspect > FIRE_ASPECT_MAX or aspect < FIRE_ASPECT_MIN: continue
        x1, y1, x2, y2 = _clip_box(x, y, x + w, y + h, width, height)
        mean_val = float(cv2.mean(hsv[y1:y2, x1:x2], mask=mask[y1:y2, x1:x2])[2])
        boxes.append({'box': (x1, y1, x2, y2), 'area': area, 'brightness': mean_val})
    return boxes, mask


def detect_smoke_regions(frame_bgr, motion_mask=None):
    height, width = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, SMOKE_LOWER, SMOKE_UPPER)
    if motion_mask is not None: mask = cv2.bitwise_and(mask, motion_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_SMOKE_AREA: continue
        x, y, w, h = cv2.boundingRect(c)
        x1, y1, x2, y2 = _clip_box(x, y, x + w, y + h, width, height)
        boxes.append({'box': (x1, y1, x2, y2), 'area': area})
    return boxes, mask


class MotionDetector:
    def __init__(self, history=300, var_threshold=32, detect_shadows=False):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=detect_shadows)

    def apply(self, frame_bgr):
        mask = self.subtractor.apply(frame_bgr)
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.medianBlur(mask, 5)
        return mask


class FlickerTracker:
    def __init__(self, history_len=15, min_samples=6, variance_threshold=35.0, match_distance=60, max_missed=3):
        self.history_len = history_len; self.min_samples = min_samples
        self.variance_threshold = variance_threshold; self.match_distance = match_distance
        # A flickering flame can drop out of the colour mask for a frame or two;
        # keep its history for a few missed frames instead of deleting it instantly.
        self.max_missed = max_missed
        self.tracks = {}; self._next_id = 0

    def _center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _find_track(self, center, exclude=()):
        best_id = None; best_dist = self.match_distance
        for track_id, track in self.tracks.items():
            # FIX: a track may be matched by at most one detection per frame.
            # Previously two nearby steady blobs fed the SAME track, mixing their
            # brightness values -> fake high variance -> false FIRE alert.
            if track_id in exclude: continue
            if not track['history']: continue
            last_center = track['last_center']
            # BUG FIXED: was "(dx * 2 + dy * 2) ** 0.5" (not Euclidean distance,
            # could even go negative under the sqrt and crash with a complex
            # number comparison). Correct Euclidean distance uses squares.
            dist = ((center[0] - last_center[0]) ** 2 + (center[1] - last_center[1]) ** 2) ** 0.5
            if dist < best_dist: best_dist = dist; best_id = track_id
        return best_id

    def update(self, detections):
        results = []; seen_ids = set()
        for det in detections:
            box = det['box']; brightness = det.get('brightness', det.get('area', 0.0))
            center = self._center(box)
            track_id = self._find_track(center, exclude=seen_ids)
            if track_id is None:
                track_id = self._next_id; self._next_id += 1
                self.tracks[track_id] = {'history': deque(maxlen=self.history_len), 'last_center': center, 'missed': 0}
            track = self.tracks[track_id]
            track['history'].append(brightness); track['last_center'] = center; track['missed'] = 0
            seen_ids.add(track_id)
            if len(track['history']) >= self.min_samples:
                values = np.array(track['history'], dtype=np.float32)
                variance = float(np.var(values)); is_flickering = variance >= self.variance_threshold
            else:
                variance = 0.0; is_flickering = False
            results.append({'box': box, 'track_id': track_id, 'flicker_variance': variance, 'is_flickering': is_flickering, 'samples': len(track['history'])})
        for tid in [t for t in self.tracks if t not in seen_ids]:
            self.tracks[tid]['missed'] += 1
            if self.tracks[tid]['missed'] > self.max_missed: del self.tracks[tid]
        return results


_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
_profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')


def has_visible_face(frame_bgr, person_box, check_profile=True):
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    upper_y2 = y1 + int(h * 0.55); upper_y2 = max(upper_y2, y1 + 1)
    crop = frame_bgr[y1:upper_y2, x1:x2]
    if crop.size == 0: return True
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY); gray = cv2.equalizeHist(gray)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=FACE_SCALE_FACTOR, minNeighbors=FACE_MIN_NEIGHBORS, minSize=FACE_MIN_SIZE)
    if len(faces) > 0: return True
    if check_profile:
        profiles = _profile_cascade.detectMultiScale(gray, scaleFactor=FACE_SCALE_FACTOR, minNeighbors=FACE_MIN_NEIGHBORS, minSize=FACE_MIN_SIZE)
        if len(profiles) > 0: return True
        flipped = cv2.flip(gray, 1)
        profiles_flipped = _profile_cascade.detectMultiScale(flipped, scaleFactor=FACE_SCALE_FACTOR, minNeighbors=FACE_MIN_NEIGHBORS, minSize=FACE_MIN_SIZE)
        if len(profiles_flipped) > 0: return True
    return False


class AlertStabilizer:
    def __init__(self, confirm_frames=5, clear_frames=8):
        self.confirm_frames = confirm_frames; self.clear_frames = clear_frames
        self.counters = defaultdict(int); self.active = defaultdict(bool); self.first_confirmed_at = {}

    def update(self, name, triggered):
        count = self.counters[name]; is_active = self.active[name]
        if triggered: count = min(count + 1, self.confirm_frames)
        else: count = max(count - self.confirm_frames / max(self.clear_frames, 1), 0)
        newly_confirmed = False
        if not is_active and count >= self.confirm_frames:
            is_active = True; newly_confirmed = True; self.first_confirmed_at[name] = time.time()
        elif is_active and count <= 0: is_active = False
        self.counters[name] = count; self.active[name] = is_active
        return is_active, newly_confirmed

    def duration_active(self, name):
        if not self.active[name] or name not in self.first_confirmed_at: return 0.0
        return time.time() - self.first_confirmed_at[name]

    def reset(self, name):
        self.counters[name] = 0; self.active[name] = False; self.first_confirmed_at.pop(name, None)


def nms_boxes(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0: return []
    boxes = np.array(boxes, dtype=np.float32); scores = np.array(scores, dtype=np.float32)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1); order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1); h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]
    return keep


def draw_label(frame, text, x, y, color, font_scale=0.6, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    y = max(y, th + 8)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)


def draw_box(frame, box, color, thickness=2):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


class SnapshotLogger:
    def __init__(self, snapshot_dir, log_path, cooldown_seconds):
        self.snapshot_dir = snapshot_dir
        self.log_path = log_path
        self.cooldown_seconds = cooldown_seconds
        self.last_saved = {}
        os.makedirs(snapshot_dir, exist_ok=True)

    def maybe_save(self, tag, frame):
        now = time.time()
        last = self.last_saved.get(tag, 0)
        if now - last < self.cooldown_seconds:
            return None
        self.last_saved[tag] = now
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path = os.path.join(self.snapshot_dir, f'{tag}_{timestamp}.jpg')
        cv2.imwrite(path, frame)
        self._log(f'{tag} alert confirmed, snapshot saved: {path}')
        return path

    def _log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f'[{timestamp}] {message}\n')
