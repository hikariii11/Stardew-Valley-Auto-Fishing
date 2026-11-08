
import pyautogui
import time

print("Screen resolution:", pyautogui.size())
print("Taking screenshot in 3 seconds...")

time.sleep(3)

image = pyautogui.screenshot()
image.save("screen_test.png")

print("Screenshot saved successfully!")
