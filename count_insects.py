#!/usr/bin/env python3
import argparse
import os
import sys
from collections import Counter
import cv2
import supervision as sv
from inference import get_model

# Ensure weights persist across reboots in offline field conditions
os.environ.setdefault("MODEL_CACHE_DIR", "models")

DEFAULT_MODEL_ID = "bugbytes-cacheusifyoucan-d0auc-4-rfdetr-nas-pecoret-t1--454bec"
DEFAULT_API_KEY = "PnGAc41GMUzYTZuCPvVW"  # Or export ROBOFLOW_API_KEY in shell


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mothbox CLI: Count target insects using sliced inference."
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the captured image file (e.g. inference_images/capture.jpg)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence threshold for detections (default: 0.40)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.getenv("ROBOFLOW_MODEL_ID", DEFAULT_MODEL_ID),
        help="Roboflow model ID with version",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("ROBOFLOW_API_KEY", DEFAULT_API_KEY),
        help="Roboflow API key",
    )
    parser.add_argument(
        "--include-other",
        action="store_true",
        help="Include the non-target 'other' class in counts",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: File not found -> {args.image}", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not decode image -> {args.image}", file=sys.stderr)
        sys.exit(1)

    h, w = image.shape[:2]

    # Initialize model (reads from local cache if pre-downloaded)
    model = get_model(model_id=args.model_id, api_key=args.api_key)

    def slice_callback(image_slice):
        results = model.infer(image_slice)[0]
        return sv.Detections.from_inference(results)

    # 4x4 tile dimensions based on native 64MP canvas (9248x6944)
    # Using 15% overlap to avoid cutting through insects straddling tile edges
    slice_w = max(640, int(w / 4))
    slice_h = max(640, int(h / 4))

    slicer = sv.InferenceSlicer(
        slice_wh=(slice_w, slice_h),
        overlap_ratio_wh=(0.15, 0.15),
        iou_threshold=0.5,
        callback=slice_callback,
    )

    detections = slicer(image)

    # Extract class names and filter by confidence threshold
    class_names = detections.data.get("class_name", [])
    confidences = detections.confidence if detections.confidence is not None else []

    counts = Counter()
    for name, conf in zip(class_names, confidences):
        if conf < args.conf:
            continue
        if not args.include_other and name == "other":
            continue
        counts[name] += 1

    total_count = sum(counts.values())

    # Formatted terminal output
    print("\n" + "=" * 32)
    print("      INSECT DETECTION SUMMARY  ")
    print("=" * 32)
    if counts:
        for insect_class, count in sorted(counts.items()):
            print(f"{insect_class:<20} : {count:>5}")
    else:
        print("No target insects detected.")
    print("-" * 32)
    print(f"{'TOTAL DETECTIONS':<20} : {total_count:>5}")
    print("=" * 32 + "\n")


if __name__ == "__main__":
    main()