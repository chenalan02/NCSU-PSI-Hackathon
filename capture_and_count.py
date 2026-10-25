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

# Suppress verbose Roboflow warnings before loading inference
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
os.environ["CORE_MODEL_YOLO_WORLD_ENABLED"] = "False"
os.environ.setdefault("MODEL_CACHE_DIR", "models")

DEFAULT_API_KEY = "PnGAc41GMUzYTZuCPvVW"
WORKSPACE_NAME = "bugbytes"
WORKFLOW_ID = "bugbytes-cacheusifyoucan-vbugbytes-cacheusifyoucan-d0auc-4-rfdetr-nas-pecoret-t1-454bec-logic"

def find_usb_drive():
    """Finds the mount point of a connected USB drive on Raspberry Pi."""
    usb_path = '/media/sda1'
    if os.path.exists(usb_path):
        return usb_path
    return '.'  # Fallback to current directory if USB is unplugged

def main():
    parser = argparse.ArgumentParser(description='Capture and count insects using local Roboflow Docker server.')

    # Camera controls
    parser.add_argument('-f', '--focus', type=str, default=None, help='Manual lens position in diopters')
    parser.add_argument('--af', type=str, choices=['auto', 'continuous', 'manual'], default=None, help="Autofocus mode")
    parser.add_argument('-t', '--timeout', type=int, default=2000, help='Warm-up time (ms)')
    parser.add_argument('--shutter', type=int, default=None, help='Shutter speed (us)')
    parser.add_argument('--name', type=str, default=None, help='Custom label tag')

    # Inference controls
    parser.add_argument('--conf', type=float, default=0.40, help='Confidence threshold')
    parser.add_argument('--api-key', type=str, default=os.getenv("ROBOFLOW_API_KEY", DEFAULT_API_KEY))
    parser.add_argument('--include-other', action='store_true', help="Include 'other' class")

    args = parser.parse_args()

    # ==========================================
    # 1. CAPTURE PHASE
    # ==========================================
    usb_path = find_usb_drive()
    if usb_path == '.':
        print('Warning: /media/sda1 not found. Saving to current directory instead.')
    else:
        print(f'USB drive found at: {usb_path}')

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
    except FileNotFoundError:
        print('rpicam-still command not found.', file=sys.stderr)
        sys.exit(1)

    # ==========================================
    # 2. INFERENCE PHASE
    # ==========================================
    image = cv2.imread(filepath)
    if image is None:
        print(f"Error: Could not decode image -> {filepath}", file=sys.stderr)
        sys.exit(1)

    print("Connecting to local Docker inference server (http://localhost:9001)...")
    
    # Initialize the client pointed at your local container
    client = InferenceHTTPClient(
        api_url="http://localhost:9001",
        api_key=args.api_key
    ).configure(InferenceConfiguration(api_key_transport="header"))

    def slice_callback(image_slice: np.ndarray) -> sv.Detections:
        """Sends a sliced tile to the local inference server and parses the Workflow response."""
        try:
            # inference_sdk automatically encodes the numpy array slice to send to the local server
            result = client.run_workflow(
                workspace_name=WORKSPACE_NAME,
                workflow_id=WORKFLOW_ID,
                images={"image": image_slice},
                use_cache=True
            )
            
            # Workflow output structure is deeply nested. Extract the predictions array.
            result_dict = result[0] if isinstance(result, list) else result
            preds = []
            
            # Search the workflow block outputs for the predictions list
            if "predictions" in result_dict:
                preds = result_dict["predictions"]
            else:
                for val in result_dict.values():
                    if isinstance(val, dict) and "predictions" in val:
                        preds = val["predictions"]
                        break
                    elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict) and "x" in val[0]:
                        preds = val
                        break

            # Convert JSON predictions back to Supervision Detections format
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

    print("Running 4x4 sliced inference...")
    h, w = image.shape[:2]
    slice_w = max(640, int(w / 4))
    slice_h = max(640, int(h / 4))

    slicer = sv.InferenceSlicer(
        slice_wh=(slice_w, slice_h),
        overlap_ratio_wh=(0.15, 0.15),
        iou_threshold=0.5,
        callback=slice_callback,
    )

    detections = slicer(image)
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