import cv2
import numpy as np

def find_arrow_tip(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define green color range (adjust if needed)
    lower_green = np.array([40, 40, 40])  # Lower bound of green in HSV
    upper_green = np.array([90, 255, 255])  # Upper bound of green in HSV

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No green arrow detected!")
        return None

    # Assume the largest contour corresponds to the arrow
    arrow_contour = max(contours, key=cv2.contourArea)

    # Find the tip of the arrow (smallest y-coordinate)
    tip_point = tuple(arrow_contour[arrow_contour[:, :, 1].argmin()][0])

    # Get the RGB value of the arrow tip
    b, g, r = img[tip_point[1], tip_point[0]]  # OpenCV stores images in BGR format
    rgb_value = (r, g, b)  # Convert BGR to RGB

    print(f"Arrow Tip Coordinates: {tip_point}")
    print(f"RGB Value at Tip: {rgb_value}")

    # Draw the detected tip on the image
    cv2.circle(img, tip_point, 5, (0, 0, 255), -1)
    cv2.imshow("Arrow Tip", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return tip_point, rgb_value

# Example usage
image_path = "C:\\SelfDrive\\2025 Map from vision\\map_img\\1738201963361403300.png"  # Replace with your image path
tip_coordinates, tip_rgb = find_arrow_tip(image_path)

