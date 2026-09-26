import os
import time
import threading
from datetime import datetime

import cv2
import torch
import torchvision

from detect_utils import (
    detect_fire_regions,
    detect_smoke_regions,
    has_visible_face,
    AlertStabilizer,
    MotionDetector,
    FlickerTracker,
    draw_label,
    draw_box,
    SnapshotLogger,
)


PERSON_LABEL_ID = 1
PERSON_CONF_THRESHOLD = 0.65   # slightly stricter -> fewer false positives
CAMERA_INDEX = 0

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

INFER_SCALE = 0.5
INFER_EVERY_N_FRAMES = 2       # will auto-adapt at runtime, see AdaptiveInferSkip

FIRE_CONFIRM_FRAMES = 6
FIRE_CLEAR_FRAMES = 10
FACE_CONFIRM_FRAMES = 5
FACE_CLEAR_FRAMES = 8
SMOKE_CONFIRM_FRAMES = 10
SMOKE_CLEAR_FRAMES = 14

FLICKER_VARIANCE_THRESHOLD = 30.0
FLICKER_MIN_SAMPLES = 6

SNAPSHOT_DIR = 'alerts'
LOG_PATH = 'alerts_log.txt'
SNAPSHOT_COOLDOWN_SECONDS = 5.0

COLOR_PERSON_OK = (0, 200, 0)
COLOR_PERSON_WARN = (0, 140, 255)
COLOR_FIRE_CANDIDATE = (0, 90, 255)
COLOR_FIRE_CONFIRMED = (0, 0, 255)
COLOR_SMOKE = (180, 180, 180)
COLOR_TEXT = (255, 255, 255)
COLOR_ALERT = (0, 0, 255)

# Target inference loop rate; if we fall behind, skip more frames automatically.
TARGET_FPS = 20.0

# If the camera delivers no new frame for this long, stop with a clear error.
CAMERA_STALL_TIMEOUT_SECONDS = 10.0
# Minimum gap between console beeps while an alert is active.
BEEP_INTERVAL_SECONDS = 1.0


class ThreadedCamera:
    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError('Could not open webcam. Check CAMERA_INDEX or camera permissions.')

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Try to force a real-time-friendly codec/FPS where the backend supports it.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.frames_captured = 0

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
                    self.frames_captured += 1
            else:
                time.sleep(0.005)

    def read(self):
        """Returns (ok, frame, frame_id). frame_id increases only when the camera
        delivers a NEW frame, so the caller can skip frames it already processed."""
        with self.lock:
            if self.frame is None:
                return False, None, 0
            return True, self.frame.copy(), self.frames_captured

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class FpsCounter:
    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self.prev_time = time.time()

    def tick(self):
        now = time.time()
        instant_fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now
        self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant_fps
        return self.fps


class AdaptiveInferSkip:
    """Automatically increases/decreases how many frames are skipped between
    person-detection inference passes, based on measured FPS, so the app stays
    responsive on slower machines and runs inference more often on fast ones."""

    def __init__(self, target_fps=TARGET_FPS, initial_every_n=INFER_EVERY_N_FRAMES,
                 min_every_n=1, max_every_n=6):
        self.target_fps = target_fps
        self.every_n = initial_every_n
        self.min_every_n = min_every_n
        self.max_every_n = max_every_n
        self._last_adjust = time.time()

    def update(self, current_fps):
        now = time.time()
        if now - self._last_adjust < 1.0:
            return self.every_n
        self._last_adjust = now

        if current_fps < self.target_fps * 0.8 and self.every_n < self.max_every_n:
            self.every_n += 1
        elif current_fps > self.target_fps * 1.15 and self.every_n > self.min_every_n:
            self.every_n -= 1
        return self.every_n

def load_person_model():
    weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights,
        box_score_thresh=PERSON_CONF_THRESHOLD,  # filter low-confidence boxes inside the model
    )
    model.eval()
    return model


def run_person_detection(model, frame_bgr, device, scale, use_half):
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device, non_blocking=True)
    if use_half:
        tensor = tensor.half()

    with torch.no_grad():
        outputs = model(tensor)[0]

    boxes = []
    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if label.item() != PERSON_LABEL_ID:
            continue
        if score.item() < PERSON_CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = box.tolist()
        x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
        boxes.append((int(x1), int(y1), int(x2), int(y2), score.item()))

    return boxes


def draw_hud(frame, fps, alerts, frame_size):
    width, height = frame_size
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)

    if alerts:
        alert_text = ' | '.join(alerts)
        cv2.putText(frame, f'ALERT: {alert_text}', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ALERT, 2)

    cv2.putText(frame, 'q: quit', (width - 90, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)


def process_fire(frame, display, flicker_tracker, stabilizer, logger):
    detections, fire_mask = detect_fire_regions(frame)
    flicker_results = flicker_tracker.update(detections)

    confirmed_boxes = []
    candidate_boxes = []

    for result in flicker_results:
        x1, y1, x2, y2 = result['box']
        if result['samples'] < FLICKER_MIN_SAMPLES:
            candidate_boxes.append(result['box'])
            draw_box(display, result['box'], COLOR_FIRE_CANDIDATE, 1)
            continue

        if result['is_flickering']:
            confirmed_boxes.append(result['box'])
            draw_box(display, result['box'], COLOR_FIRE_CONFIRMED, 2)
            draw_label(display, f"FIRE var={result['flicker_variance']:.0f}", x1, y1, COLOR_FIRE_CONFIRMED)
        else:
            candidate_boxes.append(result['box'])
            draw_box(display, result['box'], COLOR_FIRE_CANDIDATE, 1)
            draw_label(display, 'steady light', x1, y1, COLOR_FIRE_CANDIDATE)

    fire_raw = len(confirmed_boxes) > 0
    is_active, newly_confirmed = stabilizer.update('fire', fire_raw)

    if newly_confirmed or is_active:
        logger.maybe_save('fire', display)

    return is_active, fire_mask


def process_smoke(frame, display, motion_mask, stabilizer, logger):
    detections, smoke_mask = detect_smoke_regions(frame, motion_mask)

    for det in detections:
        draw_box(display, det['box'], COLOR_SMOKE, 2)
        draw_label(display, 'SMOKE?', det['box'][0], det['box'][1], COLOR_SMOKE)

    smoke_raw = len(detections) > 0
    is_active, newly_confirmed = stabilizer.update('smoke', smoke_raw)

    if newly_confirmed:
        logger.maybe_save('smoke', display)

    return is_active


def process_people(frame, display, person_model, device, use_half, last_boxes, run_inference, stabilizer, logger):
    if run_inference:
        last_boxes = run_person_detection(person_model, frame, device, INFER_SCALE, use_half)

    covered_face_raw = False
    for (x1, y1, x2, y2, score) in last_boxes:
        face_visible = has_visible_face(frame, (x1, y1, x2, y2))
        color = COLOR_PERSON_OK if face_visible else COLOR_PERSON_WARN
        draw_box(display, (x1, y1, x2, y2), color, 2)
        draw_label(display, f'person {score:.2f}', x1, y1, color)

        if not face_visible:
            covered_face_raw = True

    is_active, newly_confirmed = stabilizer.update('covered_face', covered_face_raw)

    if newly_confirmed:
        logger.maybe_save('covered_face', display)

    return last_boxes, is_active


def main():
    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, os.cpu_count() or 1))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_half = device.type == 'cuda'
    print(f'Using device: {device} (half precision: {use_half})')

    print('Loading person detection model...')
    person_model = load_person_model().to(device)
    if use_half:
        person_model = person_model.half()

    # Warm up the model so the first real frame isn't slowed down by lazy init.
    with torch.no_grad():
        dummy = torch.zeros(
            1, 3, int(CAPTURE_HEIGHT * INFER_SCALE), int(CAPTURE_WIDTH * INFER_SCALE),
            device=device
        )
        if use_half:
            dummy = dummy.half()
        person_model(dummy)

    camera = ThreadedCamera(CAMERA_INDEX, CAPTURE_WIDTH, CAPTURE_HEIGHT)
    motion_detector = MotionDetector()
    flicker_tracker = FlickerTracker(
        variance_threshold=FLICKER_VARIANCE_THRESHOLD,
        min_samples=FLICKER_MIN_SAMPLES,
    )

    fire_stabilizer = AlertStabilizer(confirm_frames=FIRE_CONFIRM_FRAMES, clear_frames=FIRE_CLEAR_FRAMES)
    face_stabilizer = AlertStabilizer(confirm_frames=FACE_CONFIRM_FRAMES, clear_frames=FACE_CLEAR_FRAMES)
    smoke_stabilizer = AlertStabilizer(confirm_frames=SMOKE_CONFIRM_FRAMES, clear_frames=SMOKE_CLEAR_FRAMES)
    logger = SnapshotLogger(SNAPSHOT_DIR, LOG_PATH, SNAPSHOT_COOLDOWN_SECONDS)
    fps_counter = FpsCounter()
    infer_skip = AdaptiveInferSkip()

    frame_idx = 0
    last_person_boxes = []
    last_frame_id = 0
    last_new_frame_time = time.time()
    last_beep_time = 0.0

    print('Press q inside the video window to quit.')

    try:
        while True:
            ok, frame, frame_id = camera.read()
            if not ok or frame is None or frame_id == last_frame_id:
                # FIX: previously the same camera frame was processed again and
                # again whenever the loop ran faster than the camera. That fed
                # duplicate samples into the flicker tracker, made alerts confirm
                # after fewer REAL frames, and polluted the motion background.
                if time.time() - last_new_frame_time > CAMERA_STALL_TIMEOUT_SECONDS:
                    print('ERROR: camera stopped delivering frames (disconnected or in use?).')
                    break
                cv2.waitKey(1)
                time.sleep(0.002)
                continue
            last_frame_id = frame_id
            last_new_frame_time = time.time()

            frame_idx += 1
            run_inference_now = (frame_idx % infer_skip.every_n == 0)

            height, width = frame.shape[:2]
            # FIX: all detectors read the clean camera frame; boxes/labels are drawn
            # on a separate copy. Before, orange/red person labels drawn earlier in
            # the same frame were seen by the fire colour detector (and boxes by the
            # face detector), corrupting detections.
            display = frame.copy()
            motion_mask = motion_detector.apply(frame)

            last_person_boxes, face_active = process_people(
                frame, display, person_model, device, use_half, last_person_boxes, run_inference_now,
                face_stabilizer, logger
            )

            fire_active, fire_mask = process_fire(frame, display, flicker_tracker, fire_stabilizer, logger)
            smoke_active = process_smoke(frame, display, motion_mask, smoke_stabilizer, logger)

            alerts = []
            if fire_active:
                alerts.append('FIRE')
            if smoke_active:
                alerts.append('SMOKE')
            if face_active:
                alerts.append('COVERED FACE')

            fps = fps_counter.tick()
            infer_skip.update(fps)
            draw_hud(display, fps, alerts, (width, height))

            cv2.imshow('Camera Security Monitor', display)
            cv2.imshow('Fire Mask', fire_mask)
            cv2.imshow('Motion Mask', motion_mask)

            # FIX: beep at most once per BEEP_INTERVAL_SECONDS instead of every frame.
            now = time.time()
            if alerts and now - last_beep_time >= BEEP_INTERVAL_SECONDS:
                print('\a', end='', flush=True)
                last_beep_time = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        # FIX: always release the camera/windows, even on Ctrl+C or an exception.
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
