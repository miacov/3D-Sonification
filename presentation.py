import cv2
import os
import pandas as pd


def apply_ripple(frames, x, y, color, velocity, thickness, draw_ring_frame=1):
    frames_lst = []
    alpha = 0.8  # transparency factor
    ring = False

    for i, frame in enumerate(frames):
        if i == draw_ring_frame:
            ring = True
        if ring:
            overlay = frame.copy()
            vel = int(velocity*alpha)
            thick = int(thickness*alpha)
            cv2.circle(overlay, (x, y), vel, color, thick)
            image_new = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            alpha -= 0.04
            if alpha <= 0:
                ring = False
            frames_lst.append(image_new)
        else:
            frames_lst.append(frame)

    return frames_lst


def process_video_frames(video_name, frame_start=0, frame_end=1000000, frame_interval=30,
                         output_processed_video_fps=30):
    # open video object
    video = cv2.VideoCapture(os.path.join("videos", video_name))
    if not video.isOpened():
        return

    # read ripple files
    file_path = os.path.join("data", video_name.replace(".mp4", ""))
    contours = pd.read_csv(os.path.join(file_path, "contour_ripples.csv"))
    lines = pd.read_csv(os.path.join(file_path, "line_ripples.csv"))
    big_celestials = pd.read_csv(os.path.join(file_path, "big_celestials_ripples.csv"))

    frame_count = 0  # actual frame counter

    frames = []

    while True:
        flag, frame = video.read()  # read frame from the video

        if frame_start <= frame_count <= frame_end:
            frames.append(frame)

        # stop if end of video is reached
        if not flag:
            break

        frame_count += 1

    prev_start = 0
    for i in range(30, len(frames), frame_interval):
        second = int(prev_start/frame_interval)

        # contour draw on star
        contour = contours.iloc[second]

        color = tuple(int(val) for val in contour["colour"].replace("(", "").replace(")", "").replace(",", " ").split())

        frames[prev_start: i] = apply_ripple(frames[prev_start: i],
                                             x=contour["x"],
                                             y=contour["y"],
                                             color=color,
                                             velocity=contour["velocity"],
                                             thickness=contour["velocity"])

        # line draw top and bottom ripples
        lines_sel = lines[lines["time"] == second]
        for idx, line in lines_sel.iterrows():
            color = tuple(
                int(val) for val in line["colour"].replace("(", "").replace(")", "").replace(",", " ").split())

            frames[prev_start: i] = apply_ripple(frames[prev_start: i],
                                                 x=int(line["x"]),
                                                 y=0,
                                                 color=color,
                                                 velocity=50,
                                                 thickness=50)

            frames[prev_start: i] = apply_ripple(frames[prev_start: i],
                                                 x=int(line["x"]),
                                                 y=1079,
                                                 color=color,
                                                 velocity=50,
                                                 thickness=50)

        # draw big celestial side ripples
        big_celestial = big_celestials.iloc[second]

        if big_celestial["velocity"] != 0:
            color = tuple(
                int(val) for val in
                big_celestial["colour"].replace("(", "").replace(")", "").replace(",", " ").split())

            frames[prev_start: i] = apply_ripple(frames[prev_start: i],
                                                 x=0,
                                                 y=540,
                                                 color=color,
                                                 velocity=big_celestial["velocity"]+20,
                                                 thickness=big_celestial["velocity"]+20)

            frames[prev_start: i] = apply_ripple(frames[prev_start: i],
                                                 x=1919,
                                                 y=540,
                                                 color=color,
                                                 velocity=big_celestial["velocity"]+20,
                                                 thickness=big_celestial["velocity"]+20)

        prev_start = i

    # release video object and close windows
    video.release()
    cv2.destroyAllWindows()

    create_video(frames, os.path.join("processed_videos", video_name.replace(".mp4", ""), "presentation.mp4"),
                 fps=output_processed_video_fps)

def create_video(frames, output_video_path, fps=1, is_color=True):
    """
    Creates video from frames.

    :param frames: frames array
    :param output_video_path: output path for video
    :param fps: frames per second
    :param is_color: video is colored if True, otherwise black and white
    """
    # get the height and width from the first frame
    if is_color:
        height, width, _ = frames[0].shape
    else:
        height, width = frames[0].shape

    # define the codec and create video object
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # mp4v, avc1, avc3 potential codecs
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=is_color)

    # write the frames to the video
    for frame in frames:
        out.write(frame)

    # release video object
    out.release()


if __name__ == "__main__":
    process_video_frames("Flight_to_AG_Carinae.mp4", frame_start=0, frame_end=10000000, frame_interval=30)
    #process_video_frames("A_Flight_to_HCG_40.mp4", frame_start=0, frame_end=10000000, frame_interval=30)