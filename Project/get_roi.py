import cv2

def get_body_roi(frame, face, verbose=False):
    h, w = frame.shape[:2]
    x, y, fw, fh = face

    face_center_x = x + fw // 2

    # Tunable parameters
    x_margin = int(5 * fw)     # sides
    body_height = int(5.0 * fh)  # downwards

    x1 = max(0, face_center_x - x_margin)
    x2 = min(w, face_center_x + x_margin)
    y1 = max(0, y)
    y2 = h

    if verbose:
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    return x1, y1, x2, y2
