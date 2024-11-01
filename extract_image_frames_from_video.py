import cv2
import numpy as np
from pathlib import Path

# path to the video
video_path = Path("results") / "ns-2" / "sinusoidal_actuation" / "ns-2_task_space_animation_noise_comp.mp4"

# dt between saved image frames
duration = 7.0
video_speedup_factor = duration / 23.3
save_img_dt = duration / 5
print(f"Saving image every {save_img_dt} seconds")

# crop margin
crop_margin = [10, 10, 25, 60]  # [top, bottom, left, right]

if __name__ == "__main__":
    # read the video
    cap = cv2.VideoCapture(str(Path(video_path).expanduser()))
    # extract the frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video frame rate: {fps}")

    # iterate over the frames
    time_last_saved = -np.inf
    while cap.isOpened():
        # current time in the video
        time_current = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 * video_speedup_factor
        ret, frame = cap.read()
        if not ret:
            break

        # calculate the time since the last saved frame
        time_since_last_saved = time_current - time_last_saved
        print(
            f"Current time: {time_current:.2f}, Time since last saved: {time_since_last_saved:.2f}"
        )
        if time_since_last_saved >= (save_img_dt - 1e-6) or np.abs(time_current - duration) < 1e-2:
            img_path = video_path.parent / f"{video_path.stem}_{time_current:.2f}.png"

            # crop the image
            frame = frame[
                crop_margin[0] : frame.shape[0] - crop_margin[1],
                crop_margin[2] : frame.shape[1] - crop_margin[3],
            ]

            # save the frame
            print(f"Saving frame of t={time_current} to {img_path.resolve()}")
            cv2.imwrite(str(img_path), frame)
            time_last_saved = time_current