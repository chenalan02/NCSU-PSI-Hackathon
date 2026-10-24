import os
import subprocess
from datetime import datetime

def find_usb_drive():
    """Finds the mount point of a connected USB drive on Raspberry Pi."""
    # Based on your df output, your drive mounts directly here
    usb_path = "/media/sda1"
    
    if os.path.exists(usb_path):
        return usb_path
    return "." # Fallback to current directory if USB is unplugged

def take_photo():
    print("Searching for connected USB drive...")
    usb_path = find_usb_drive()
    
    if usb_path == ".":
        print("Warning: /media/sda1 not found. Saving to current directory instead.")
    else:
        print(f"USB drive found at: {usb_path}")

    # Generate filename with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mothbox_test_{timestamp}.jpg"
    filepath = os.path.join(usb_path, filename)
    
    print(f"Warming up camera and capturing to: {filepath}")
    
    # List of camera commands to try (Newest OS -> Older OS -> Oldest OS)
    commands_to_try = [
        ["rpicam-still", "-t", "1500", "--autofocus", "-o", filepath],
        ["libcamera-still", "-t", "1500", "--autofocus", "-o", filepath],
        ["raspistill", "-t", "1500", "-o", filepath]
    ]
    
    success = False
    for cmd in commands_to_try:
        try:
            # Hide the massive amount of terminal text the camera command usually prints
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Success! Photo saved as {filename} using {cmd[0]}")
            success = True
            break
        except FileNotFoundError:
            # Command doesn't exist on this OS version, try the next one
            continue
        except subprocess.CalledProcessError:
            print(f"\nError: {cmd[0]} crashed. Is the ribbon cable connected securely?")
            success = True # We found the command, but it failed. Stop trying.
            break

    if not success:
        print("\nFailed: No camera commands found. Is the camera enabled?")

if __name__ == "__main__":
    take_photo()