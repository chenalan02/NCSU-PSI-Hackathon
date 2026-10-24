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
      if args.focus is not None
      else f'af_{args.af}'
      if args.af
      else 'default'
  )
  label_tag = f'_{args.name}' if args.name else ''
  filename = f'moth_{timestamp}_{focus_tag}{label_tag}.jpg'
  filepath = os.path.join(usb_path, filename)

  # Build camera command
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
    # Ensure USB cache is flushed so image writes safely
    subprocess.run(['sync'])
    print(f'Success! Photo saved to: {filepath}')
  except subprocess.CalledProcessError as e:
    print(f'Capture failed with exit code: {e.returncode}')
  except FileNotFoundError:
    print('rpicam-still command not found.')


if __name__ == '__main__':
  main()