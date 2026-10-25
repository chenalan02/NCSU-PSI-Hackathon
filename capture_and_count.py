#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime
from collections import Counter
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
    parser = argparse.ArgumentParser(description='Capture and count insects using Roboflow Cloud API.')
    parser.add_argument('-f', '--focus', type=str, default=None)
    parser.add_argument('--af', type=str, choices=['auto', 'continuous', 'manual'], default=None)
    parser.add_argument('-t', '--timeout', type=int, default=2000)
    parser.add_argument('--shutter', type=int, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--conf', type=float, default=0.45) # Bumped slightly to fix the extra chinch bug
    parser.add_argument('--api-key', type=str, default=os.getenv("ROBOFLOW_API_KEY", DEFAULT_API_KEY))
    parser.add_argument('--include-other', action='store_true')

    args = parser.parse_args()

    # 1. CAPTURE
    usb_path = find_usb_drive()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    focus_tag = f'lens_{args.focus}' if args.focus is not None else f'af_{args.af}' if args.af else 'default'
    label_tag = f'_{args.name}' if args.name else ''
    filename = f'moth_{timestamp}_{focus_tag}{label_tag}.jpg'
    filepath = os.path.join(usb_path, filename)

    cmd = ['rpicam-still', '-t', str(args.timeout), '-o', filepath]
    if args.focus is not None: cmd.extend(['--lens-position', str(args.focus)])
    elif args.af is not None: cmd.extend(['--autofocus-mode', args.af])
    if args.shutter is not None: cmd.extend(['--shutter', str(args.shutter)])

    print(f'Running command: {" ".join(cmd)}')
    try:
        subprocess.run(cmd, check=True)
        subprocess.run(['sync'])
        print(f'Success! Photo saved to: {filepath}')
    except subprocess.CalledProcessError as e:
        print(f'Capture failed: {e.returncode}', file=sys.stderr)
        sys.exit(1)

    # 2. CLOUD INFERENCE (Server handles the 4x4 tiling automatically)
    print("Sending full image to Roboflow Cloud (Server-side 4x4 Tiling)...")
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=args.api_key
    ).configure(InferenceConfiguration(api_key_transport="header"))

    try:
        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": filepath}
        )
    except Exception as e:
        print(f"Cloud inference failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. COUNT
    counts = Counter()
    try:
        predictions = result[0]['predictions']['predictions']
    except (IndexError, KeyError, TypeError):
        predictions = []

    for p in predictions:
        conf = p.get('confidence', 0)
        name = p.get('class', 'unknown')

        if conf < args.conf:
            continue
        if not args.include_other and name == "other":
            continue
            
        counts[name] += 1

    # 4. OUTPUT
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