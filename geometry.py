
import cv2
import numpy as np


def detect_shape(approx):

    vertices = len(approx)

    if vertices == 3:
        return "Triangle"

    elif vertices == 4:

        rect = cv2.minAreaRect(approx)

        w, h = rect[1]

        if min(w, h) == 0:
            return "Unknown"

        ratio = max(w, h) / min(w, h)

        if ratio <= 1.10:
            return "Square"

        else:
            return "Rectangle"

    elif vertices == 5:
        return "Pentagon"

    elif vertices == 6:
        return "Hexagon"

    else:

        perimeter = cv2.arcLength(approx, True)
        area = cv2.contourArea(approx)

        if perimeter == 0:
            return "Unknown"

        circularity = (4 * np.pi * area) / (perimeter ** 2)

        if circularity > 0.80:
            return "Circle"

        return "Other"


def analyze_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result_image = image.copy()

    results = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 500:
            continue

        perimeter = cv2.arcLength(contour, True)

        epsilon = 0.02 * perimeter

        approx = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        shape = detect_shape(approx)

        x, y, w, h = cv2.boundingRect(approx)

        M = cv2.moments(approx)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = x + w // 2
            cy = y + h // 2

        cv2.drawContours(
            result_image,
            [approx],
            -1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            result_image,
            shape,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        results.append({
            "shape": shape,
            "area": round(area, 2),
            "perimeter": round(perimeter, 2),
            "center": (cx, cy)
        })

    return result_image, results
