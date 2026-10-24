import os
import subprocess
from datetime import datetime

def find_usb_drive():
    """Finds the mount point of a connected USB drive on Raspberry Pi."""
    # Pi OS auto-mounts USBs under /media/<username>/<usb_name>
    media_dir = "/media"
    
    if os.path.exists(media_dir):
        # Look through user directories in /media (e.g., /media/pi or /media/mothbox)
        users = os.listdir(media_dir)
        for user in users:
            user_path = os.path.join(media_dir, user)
            if os.path.isdir(user_path):
                drives = os.listdir(user_path)
                if drives:
                    # Return the path to the first USB drive found
                    return os.path.join(user_path, drives[0])
                    
    return None

def take_photo():
    print("Searching for connected USB drive...")
    usb_path = find_usb_drive()
    
    if not usb_path:
        print("Error: No USB drive found in /media/. Saving to current directory instead.")
        usb_path = "." 
    else:
        print(f"USB drive found at: {usb_path}")

    # Generate filename with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mothbox_test_{timestamp}.jpg"
    filepath = os.path.join(usb_path, filename)
    
    print(f"Warming up camera and capturing to: {filepath}")
    
    # libcamera-still is the standard command for modern Pi OS
    # -t 1500 gives the sensor 1.5 seconds to adjust focus and auto-exposure
    # --autofocus is useful if you are using an Arducam or Pi Cam v3 with AF
    command = ["libcamera-still", "-t", "1500", "--autofocus", "-o", filepath]
    
    try:
        subprocess.run(command, check=True)
        print(f"Success! Photo saved as {filename}")
    except subprocess.CalledProcessError:
        print("\nFailed with libcamera. Falling back to legacy raspistill...")
        # Fallback for older Raspberry Pi OS versions (Buster and older)
        try:
            legacy_cmd = ["raspistill", "-t", "1500", "-o", filepath]
            subprocess.run(legacy_cmd, check=True)
            print(f"Success! Photo saved as {filename}")
        except Exception as e:
            print(f"Failed to capture photo: {e}")
    except FileNotFoundError:
        print("Camera commands not found. Is the camera enabled in raspi-config?")

if __name__ == "__main__":
    take_photo()