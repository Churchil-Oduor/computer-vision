import numpy as np
import cv2 as cv

class Render:
    def __init__(self, alpha=0.3):
        self.p_smoothed_z = 1
        self.alpha = alpha

    def overlayImg(self, frame, png, position, scale=1):
        new_w = int(png.shape[1] * scale * 0.2)
        new_h = int(png.shape[0] * scale * 0.2)

        resized = cv.resize(png, (new_w, new_h), interpolation=cv.INTER_AREA)

        x, y = position
        x -= new_w // 2
        y -= new_h // 2

        if x < 0: x = 0
        if y < 0: y = 0

        if x + new_w > frame.shape[1]:
            new_w = frame.shape[1] - x
            resized = resized[:, :new_w]

        if y + new_h > frame.shape[0]:
            new_h = frame.shape[0] - y
            resized = resized[:new_h, :]

        if resized.shape[2] == 4:
            overlay = resized[:, :, :3]
            mask = resized[:, :, 3] / 255.0
            mask = mask[..., np.newaxis]
        else:
            overlay = resized
            mask = np.ones((resized.shape[0], resized.shape[1], 1))

        roi = frame[y:y+new_h, x:x+new_w]

        # Blend the overlay with the frame ROI using the mask
        roi[:] = (1.0 - mask) * roi + mask * overlay

        return frame

    def lowPassFilter(self, current_z):
        smoothed_z = self.alpha * current_z + (1 - self.alpha) * self.p_smoothed_z
        self.p_smoother_z = smoothed_z
        return smoothed_z
