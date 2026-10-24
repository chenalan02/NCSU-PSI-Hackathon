import argparse
import os
import subprocess
from datetime import datetime


def find_usb_drive():
  """Finds the mount point of a connected USB drive on Raspberry Pi."""
  usb_path = '/media/sda1'
  if os.path.exists(usb_path):
    return usb_path
  return '.'  # Fallback to current directory if USB is unplugged


def main():
  parser = argparse.ArgumentParser(
      description='Capture test photos with customizable focus parameters.'
  )

  # Focus controls
  parser.add_argument(
      '-f',
      '--focus',
      type=str,
      default=None,
      help=(
          'Manual lens position in diopters (e.g., 7.0 for ~14cm macro, 1.0 for'
          ' ~1m, 0.0 for infinity).'
      ),
  )
  parser.add_argument(
      '--af',
      type=str,
      choices=['auto', 'continuous', 'manual'],
      default=None,
      help="Autofocus mode: 'auto', 'continuous', or 'manual'.",
  )

  # Capture tuning
  parser.add_argument(
      '-t',
      '--timeout',
      type=int,
      default=2000,
      help=(
          'Warm-up/preview time in milliseconds before shutter fires (default:'
          ' 2000).'
      ),
  )
  parser.add_argument(
      '--shutter',
      type=int,
      default=None,
      help='Explicit shutter speed / exposure time in microseconds (us).',
  )
  parser.add_argument(
      '--name',
      type=str,
      default=None,
      help=(
          'Custom label tag to add to the output filename (e.g.,'
          ' "macro_run1").'
      ),
  )

  args = parser.parse_args()

  usb_path = find_usb_drive()
  if usb_path == '.':
    print('Warning: /media/sda1 not found. Saving to current directory instead.')
  else:
    print(f'USB drive found at: {usb_path}')

  # Build descriptive filename
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  focus_tag = (
      f'lens_{args.focus}'
      if args.focus is