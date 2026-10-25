#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime
from collections import Counter
import numpy as np
import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

DEFAULT_API_KEY = "PnGAc41GMUzYTZuCPvVW"
WORKSPACE_NAME = "bugbytes"
WORKFLOW_ID = "bugbytes-cacheusifyoucan-vbugbytes-cacheusifyoucan-d0auc-4-rfdetr-nas-pecoret-t1-454bec-logic"

def find_usb_drive():
    usb_path = '/media/sda1'
    if os.path.exists(usb_path):
        return usb_path
    return '.'

def main():
    parser = argparse.ArgumentParser(description='Capture and count insects using Cloud API with Image Slicing.')

    parser.add_argument('-f', '--focus', type=str, default=None)
    parser.add_argument('--af', type=str, choices=['auto', 'continuous', 'manual'], default=None)
    parser.add_argument('-t', '--timeout', type=int, default=2000)
    parser.add_argument('--shutter', type=int, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--conf', type=float, default=0.40)
    parser.add_argument('--api-key', type=str, default=os.getenv("ROBOFLOW_API_KEY", DEFAULT_API_KEY))
    parser.add_argument('--include-other', action='store_true')

    args = parser.parse_args()

    # ==========================================
    # 1. CAPTURE PHASE
    # ==========================================
    usb_path = find_usb_drive()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    focus_tag = f'lens_{args.focus}' if args.focus is not None else f'af_{args.af}' if args.af else 'default'
    label_tag = f'_{args.name}' if args.name else ''
    filename = f'moth_{timestamp}_{focus_tag}{label_tag}.jpg'
    filepath = os.path.join(usb_path, filename)

    cmd = ['rpicam-still', '-t', str(args.timeout), '-o', filepath]
    if args.focus is not None:
        cmd.extend(['--lens-position', str(args.focus)])
    elif args.af is not None:
        cmd.extend(['--autofocus-mode', args.af])
    if args.shutter is not None:
        cmd.extend(['--shutter', str(args.shutter)])

    print(f'Running command: {" ".join(cmd)}')
    try:
        subprocess.run(cmd, check=True)
        subprocess.run(['sync'])
        print(f'Success! Photo saved to: {filepath}')
    except subprocess.CalledProcessError as e:
        print(f'Capture failed with exit code: {e.returncode}', file=sys.stderr)
        sys.exit(1)

    # ==========================================
    # 2. INFERENCE PHASE (SLICED CLOUD API)
    # ==========================================
    image = cv2.imread(filepath)
    if image is None:
        print(f"Error: Could not decode image -> {filepath}", file=sys.stderr)
        sys.exit(1)

    print("Connecting to Roboflow Cloud API...")
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=args.api_key
    ).configure(InferenceConfiguration(api_key_transport="header"))

    def slice_callback(image_slice: np.ndarray) -> sv.Detections:
        """Sends a high-res slice to the cloud and parses the JSON response."""
        try:
            result = client.run_workflow(
                workspace_name=WORKSPACE_NAME,
                workflow_id=WORKFLOW_ID,
                images={"image": image_slice}
            )
            
            # Extract predictions from the nested Workflow JSON
            try:
                preds = result[0]['predictions']['predictions']
            except (IndexError, KeyError, TypeError):
                preds = []

            # Convert JSON back into supervision format
            xyxy, confidence, class_id, class_name = [], [], [], []
            for p in preds:
                cx, cy, w, h = p['x'], p['y'], p['width'], p['height']
                xyxy.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
                confidence.append(p.get('confidence', 1.0))
                class_id.append(p.get('class_id', 0))
                class_name.append(p.get('class', 'unknown'))

            if not xyxy:
                return sv.Detections.empty()

            return sv.Detections(
                xyxy=np.array(xyxy),
                confidence=np.array(confidence),
                class_id=np.array(class_id),
                data={"class_name": np.array(class_name)}
            )

        except Exception as e:
            print(f"Slice inference failed: {e}", file=sys.stderr)
            return sv.Detections.empty()

    print("Running 4x4 sliced inference on the Cloud...")
    h, w = image.shape[:2]
    slice_w = max(640, int(w / 4))
    slice_h = max(640, int(h / 4))

    # Re-introduce the slicer to handle the 9K resolution properly
    slicer = sv.InferenceSlicer(
        slice_wh=(slice_w, slice_h),
        overlap_wh=(int(slice_w * 0.15), int(slice_h * 0.15)),
        iou_threshold=0.35,
        callback=slice_callback,
    )

    detections = slicer(image)
    
    # Extract merged results
    class_names = detections.data.get("class_name", [])
    confidences = detections.confidence if detections.confidence is not None else []

    counts = Counter()
    for name, conf in zip(class_names, confidences):
        if conf < args.conf:
            continue
        if not args.include_other and name == "other":
            continue
        counts[name] += 1

    # ==========================================
    # 3. OUTPUT SUMMARY
    # ==========================================
    print("\n" + "=" * 32)
    print("      INSECT DETECTION SUMMARY  ")
    print("=" * 32)
    print(f"Image  : {filename}")
    print("-" * 32)
    if counts:
        for insect_class, count in sorted(counts.items()):
            print(f"{insect_class:<20} : {count:>5}")
    else:
        print("No target insects detected.")
    print("-" * 32)
    print(f"{'TOTAL DETECTIONS':<20} : {sum(counts.values()):>5}")
    print("=" * 32 + "\n")

if __name__ == '__main__':
    main()