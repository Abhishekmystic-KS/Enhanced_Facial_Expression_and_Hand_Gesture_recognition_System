import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_prediction_text(image, text, position, font_scale=0.7, thickness=2, text_color=(0, 255, 0), bg_color=(0, 0, 0)):
    """
    Draw text with background on the image.

    Args:
        image: The input image.
        text: Text to be drawn.
        position: Position (x, y) to draw the text.
        font_scale: Font scale.
        thickness: Thickness of the text.
        text_color: Color of the text.
        bg_color: Color of the background rectangle.

    Returns:
        image: Image with text drawn.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Calculate background rectangle dimensions
    rect_width = text_size[0] + 10
    rect_height = text_size[1] + 10

    # Calculate background rectangle coordinates
    rect_x = position[0]
    rect_y = position[1] - text_size[1] - 5

    # Draw background rectangle
    cv2.rectangle(image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), bg_color, -1)

    # Draw text
    cv2.putText(image, text, (position[0] + 5, position[1] - 5), font, font_scale, text_color, thickness)

    return image

def draw_confidence_bar(image, confidence, position, width=100, height=20, color_low=(0, 0, 255), color_high=(0, 255, 0)):
    """
    Draw a confidence bar on the image.

    Args:
        image: The input image.
        confidence: Confidence value (0.0 to 1.0).
        position: Position (x, y) to draw the bar.
        width: Width of the bar.
        height: Height of the bar.
        color_low: Color for low confidence.
        color_high: Color for high confidence.

    Returns:
        image: Image with confidence bar drawn.
    """
    # Ensure confidence is between 0 and 1
    confidence = max(0, min(1, confidence))

    # Calculate bar dimensions
    filled_width = int(width * confidence)

    # Calculate color based on confidence
    r = int(color_low[0] * (1 - confidence) + color_high[0] * confidence)
    g = int(color_low[1] * (1 - confidence) + color_high[1] * confidence)
    b = int(color_low[2] * (1 - confidence) + color_high[2] * confidence)
    color = (b, g, r)  # OpenCV uses BGR

    # Draw background (empty) bar
    cv2.rectangle(image, position, (position[0] + width, position[1] + height), (50, 50, 50), -1)

    # Draw filled bar
    cv2.rectangle(image, position, (position[0] + filled_width, position[1] + height), color, -1)

    # Draw border
    cv2.rectangle(image, position, (position[0] + width, position[1] + height), (200, 200, 200), 1)

    # Draw confidence percentage
    percentage_text = f"{int(confidence * 100)}%"
    text_position = (position[0] + width // 2 - 15, position[1] + height // 2 + 5)
    cv2.putText(image, percentage_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return image

def create_prediction_overlay(image, face_expression=None, face_confidence=None, hand_gesture=None, hand_confidence=None, hand_gestures=None):
    """
    Create an overlay with prediction results.

    Args:
        image: The input image.
        face_expression: Predicted facial expression.
        face_confidence: Confidence of facial expression prediction.
        hand_gesture: Predicted hand gesture (for backward compatibility).
        hand_confidence: Confidence of hand gesture prediction (for backward compatibility).
        hand_gestures: List of dictionaries containing gesture information for multiple hands.
            Each dictionary should have 'gesture' and 'confidence' keys.

    Returns:
        image: Image with prediction overlay.
    """
    # Create a copy of the image
    overlay = image.copy()

    # Draw facial expression prediction
    if face_expression is not None and face_confidence is not None:
        # Draw text
        overlay = draw_prediction_text(
            overlay,
            f"Expression: {face_expression}",
            (10, 30)
        )

        # Draw confidence bar
        overlay = draw_confidence_bar(
            overlay,
            face_confidence,
            (10, 40),
            width=150
        )

    # Draw hand gesture predictions for multiple hands
    if hand_gestures and len(hand_gestures) > 0:
        y_position = 100
        for i, hand_info in enumerate(hand_gestures):
            if 'gesture' in hand_info and 'confidence' in hand_info:
                # Determine hand label (left/right if available)
                hand_label = f"Hand {i+1}"
                if 'handedness' in hand_info:
                    hand_label = hand_info['handedness']

                # Draw text
                overlay = draw_prediction_text(
                    overlay,
                    f"{hand_label}: {hand_info['gesture']}",
                    (10, y_position)
                )

                # Draw confidence bar
                overlay = draw_confidence_bar(
                    overlay,
                    hand_info['confidence'],
                    (10, y_position + 10),
                    width=150
                )

                # Increment y position for next hand
                y_position += 70
    # For backward compatibility
    elif hand_gesture is not None and hand_confidence is not None:
        # Draw text
        overlay = draw_prediction_text(
            overlay,
            f"Gesture: {hand_gesture}",
            (10, 100)
        )

        # Draw confidence bar
        overlay = draw_confidence_bar(
            overlay,
            hand_confidence,
            (10, 110),
            width=150
        )

    return overlay
