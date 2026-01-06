import mimetypes
import winreg
import sys

def fix_js_mime_type():
    try:
        # Define the registry path for .js extension
        key_path = r"CLASSES_ROOT\.js"
        
        # Open the registry key
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Classes\.js", 0, winreg.KEY_SET_VALUE) as key:
            # Set the Content Type to application/javascript
            winreg.SetValueEx(key, "Content Type", 0, winreg.REG_SZ, "application/javascript")
            print("Successfully updated HKLM\\SOFTWARE\\Classes\\.js Content Type to application/javascript")
            
    except PermissionError:
        print("Permission denied. Please run this script as Administrator.")
        # Try HKEY_CURRENT_USER as a fallback if HKLM fails
        try:
             with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\.js") as key:
                winreg.SetValueEx(key, "Content Type", 0, winreg.REG_SZ, "application/javascript")
                print("Successfully updated HKCU\\Software\\Classes\\.js Content Type to application/javascript")
        except Exception as e:
            print(f"Failed to update HKCU: {e}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print(f"Current MIME type for .js: {mimetypes.guess_type('test.js')[0]}")
    fix_js_mime_type()
    # Re-initialize mimetypes to check the change
    mimetypes.init()
    print(f"New MIME type for .js: {mimetypes.guess_type('test.js')[0]}")
