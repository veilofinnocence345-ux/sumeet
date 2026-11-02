"""Scanner: identify and count objects on-device (YOLOv8 + OpenCV).

Live camera:   python scanner.py
Single image:  python scanner.py --image photo.jpg

Keys (live): SPACE scan / scan again   L live counting on/off   Q quit
"""
import argparse
import sys
from collections import Counter

import cv2
from ultralytics import YOLO

WINDOW = "Scanner"
INK = (17, 17, 17)
PAPER = (250, 250, 250)


def detect(model, frame, conf):
    """Return a list of (label, confidence, (x1, y1, x2, y2))."""
    result = model(frame, conf=conf, verbose=False)[0]
    dets = []
    for box in result.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        label = model.names[int(box.cls[0])]
        dets.append((label, float(box.conf[0]), (x1, y1, x2, y2)))
    return dets


def tally(dets):
    return Counter(label for label, _, _ in dets).most_common()


def draw_boxes(frame, dets):
    out = frame.copy()
    for label, score, (x1, y1, x2, y2) in dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), PAPER, 2)
        text = f"{label} {score:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y1 - th - 8, 0)
        cv2.rectangle(out, (x1, top), (x1 + tw + 8, top + th + 8), PAPER, -1)
        cv2.putText(out, text, (x1 + 4, top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
    return out


def draw_panel(frame, counts, hint):
    """Minimal overlay: total, per-object counts, one hint line."""
    out = frame.copy()
    total = sum(n for _, n in counts)
    lines = []
    if counts:
        lines.append(f"{total} object{'s' if total != 1 else ''}")
        lines += [f"{name}  {n}" for name, n in counts[:8]]
    lines.append(hint)

    pad, lh = 12, 24
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
                for t in lines) + pad * 2
    height = lh * len(lines) + pad
    overlay = out.copy()
    cv2.rectangle(overlay, (10, 10), (10 + width, 10 + height), INK, -1)
    out = cv2.addWeighted(overlay, 0.7, out, 0.3, 0)
    y = 10 + pad + 8
    for i, text in enumerate(lines):
        big = counts and i == 0
        cv2.putText(out, text, (10 + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 if big else 0.6,
                    PAPER, 2 if big else 1, cv2.LINE_AA)
        y += lh
    return out


def run_image(model, path, conf):
    frame = cv2.imread(path)
    if frame is None:
        sys.exit("Can't read that image.")
    dets = detect(model, frame, conf)
    counts = tally(dets)
    if not counts:
        print("Nothing found. Try more light or move closer.")
        return
    print(f"{sum(n for _, n in counts)} objects")
    for name, n in counts:
        print(f"  {name}  {n}")
    out_path = "scanned.jpg"
    cv2.imwrite(out_path, draw_panel(draw_boxes(frame, dets), counts, ""))
    print(f"Saved {out_path}")


def run_live(model, cam_index, conf):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        sys.exit("Camera off. Enable it in Settings.")

    frozen = None  # (annotated frame) while a scan result is shown
    live_count = False
    print("Point the camera at objects to count them.")

    while True:
        ok, frame = cap.read()
        if not ok:
            sys.exit("Camera off. Enable it in Settings.")

        if frozen is not None:
            view = frozen
        elif live_count:
            dets = detect(model, frame, conf)
            view = draw_panel(draw_boxes(frame, dets), tally(dets),
                              "SPACE scan   L live off   Q quit")
        else:
            view = draw_panel(frame, [], "SPACE scan   L live on   Q quit")

        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("l"):
            live_count = not live_count
            frozen = None
        if key == 32:  # SPACE
            if frozen is not None:
                frozen = None
                continue
            dets = detect(model, frame, conf)
            counts = tally(dets)
            if counts:
                frozen = draw_panel(draw_boxes(frame, dets), counts,
                                    "SPACE scan again   Q quit")
            else:
                frozen = draw_panel(frame, [],
                                    "Nothing found. Try more light.")

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Identify and count objects.")
    ap.add_argument("--image", help="scan one image instead of the camera")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="minimum confidence (0-1)")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="YOLO weights (yolov8n.pt fast, yolov8m.pt accurate)")
    args = ap.parse_args()

    model = YOLO(args.model)  # downloads weights once, then works offline
    if args.image:
        run_image(model, args.image, args.conf)
    else:
        run_live(model, args.camera, args.conf)


if __name__ == "__main__":
    main()
