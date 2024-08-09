import numpy as np
import cv2 as cv

from golf_ball_detection.postprocess.nms import heatmap_nms
from golf_ball_detection.postprocess.utils import peak_mask_to_points

IMSHOW_DELAY = 1

TARGET_WIDTH = 270
TARGET_HEIGHT = 480

VIDEO_ID = "01"
HEATMAP_VIDEO_PATH = f"/home/james/Desktop/IdeasLab/Heatmap/Heatmap/{VIDEO_ID}.mp4"
SOURCE_VIDEO_PATH = f"/home/james/Desktop/IdeasLab/Heatmap/Videos/{VIDEO_ID}.MOV"


def get_frame(cap, target_width, target_height, normalize=False):
    """Get a frame and resize to the given size from a VideoCapture."""

    # Get a frame
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Can't get frame from video")

    # Normalize
    if normalize:
        frame = frame / 255.0

    # Resize
    frame = cv.resize(
        frame, (target_width, target_height), interpolation=cv.INTER_LINEAR
    )

    return frame


if __name__ == "__main__":
    # Open source videos
    source_cap = cv.VideoCapture(SOURCE_VIDEO_PATH)
    heatmap_cap = cv.VideoCapture(HEATMAP_VIDEO_PATH)

    while source_cap.isOpened():
        # Load video frames
        try:
            source = get_frame(source_cap, TARGET_WIDTH, TARGET_HEIGHT)
            heatmap = get_frame(heatmap_cap, TARGET_WIDTH, TARGET_HEIGHT)
        except RuntimeError as e:
            print(e)
            break

        heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2GRAY)

        # Post processing
        peak_mask = heatmap_nms(heatmap, 0.3, 15)
        detections = peak_mask_to_points(peak_mask)

        # Draw and show detections
        for coord in detections:
            cv.circle(source, coord, 5, (255, 0, 0), 2)

        cv.imshow("frame", source)
        if cv.waitKey(IMSHOW_DELAY) == 27:
            break

    cv.destroyAllWindows()
