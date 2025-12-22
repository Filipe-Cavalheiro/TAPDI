import cv2 as cv

def down_sampling(frame, crop_size=368):
    """
    Downsamples an image to a square of size crop_size x crop_size.

    Parameters:
        frame (np.array): Input image of any size.
        crop_size (int): Desired output size (square).

    Returns:
        np.array: Downsampled square image.
    """
    h, w = frame.shape[:2]

    # Determine scale to preserve aspect ratio
    scale = crop_size / min(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize while keeping aspect ratio
    resized = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)

    # Center crop to crop_size x crop_size
    start_x = (new_w - crop_size) // 2
    start_y = (new_h - crop_size) // 2
    cropped = resized[start_y:start_y+crop_size, start_x:start_x+crop_size]

    return cropped
