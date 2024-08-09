import numpy as np
import cv2 as cv


def heatmap_nms(heatmap, conf_threshold, kernel_size):
    """Perform NMS on the given heatmap and return the mask of the peak pixels."""

    # Heatmap dilation (max pooling)
    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (2 * kernel_size + 1, 2 * kernel_size + 1),
        (kernel_size, kernel_size),
    )
    dilated_heatmap = cv.dilate(heatmap, kernel)

    # Generate a mask of peak pixels
    peak_mask = np.logical_and(heatmap > conf_threshold, heatmap == dilated_heatmap)

    return peak_mask
