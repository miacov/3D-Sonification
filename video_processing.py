import cv2
import os
import numpy as np
import pandas as pd
import shutil


def black_and_white_video_frame(frame_color, black_and_white_threshold=26):
    """
    Returns black and white image from colored image with a certain threshold for values.

    :param frame_color: colored image
    :param black_and_white_threshold: max value for black values, rest is white
    :return: returns black and white image
    """
    # convert to grayscale
    frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

    # convert to only black and white values
    frame_bw = np.array(frame_gray)
    frame_bw[frame_bw <= black_and_white_threshold] = 0
    frame_bw[frame_bw > black_and_white_threshold] = 255

    # print tests
    # cv2.imshow("Original image", frame_color)
    # cv2.imshow("Gray image", frame_gray)
    # cv2.imshow("B&W image", frame_bw)
    # cv2.waitKey(0)

    # cv2.imwrite("color.png" , frame_color)
    # cv2.imwrite("gray.png" , frame_gray)
    # cv2.imwrite("b&w.png" , frame_bw)

    return frame_bw


def get_contour_color(contour, frame_color):
    """
    Finds median color for a contour.

    :param contour: contour object
    :param frame_color: image with color to get contour color from
    :return: returns median color for contour
    """
    # create an empty mask
    contour_mask = np.zeros(frame_color.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # use the mask to extract the region within the contour
    contour_roi = cv2.bitwise_and(frame_color, frame_color, mask=contour_mask)

    # mask non-zero pixels to create an array of colors within the contour
    colors_within_contour = contour_roi[np.where(contour_mask > 0)]

    if colors_within_contour.size > 0:
        # get median color from colors
        return np.median(colors_within_contour, axis=0).astype(int)
    else:
        return np.array((-1, -1, -1))


def check_contour_line_overlap(contour, line_positions, frame, frame_draw=None):
    """
    Checks if contour overlaps with vertical lines across an image.

    :param contour: contour object
    :param line_positions: array with start and end x positions of each line
    :param frame: black and white frame
    :param frame_draw: optional image to draw the touching contours on (new image if none is given)
    :return: returns the indexes of the lines that overlap with the contour from array of line positions
    """
    if frame_draw is None:
        frame_draw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # create an empty mask
    contour_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # use the mask to extract the region within the contour
    contour_roi = cv2.bitwise_and(frame, frame, mask=contour_mask)

    # check if the region within the contour and the given rectangle intersect
    touch_indexes = []
    for idx, line_position in enumerate(line_positions):
        if np.any(contour_roi[:, line_position[0]:line_position[1]] == 255):
            touch_indexes.append(idx)

            # draw touching contour with blue color
            moments = cv2.moments(contour)
            # centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            frame_draw = cv2.circle(frame_draw, (line_position[0], centroid_y), 10, (255, 0, 0), -1)

    return touch_indexes


def get_contours(frame, frame_draw=None):
    """
    Generates contours on an image.

    :param frame: black and white image to get contours out of
    :param frame_draw: optional image to draw the contours on (new image if none is given)
    :return: returns contour objects and image with contours drawn on it
    """
    if frame_draw is None:
        frame_draw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # create contours from black and white image
    masked_array = np.ma.masked_where(frame > 255, frame)
    contours, hierarchy = cv2.findContours(masked_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame_draw, contours, -1, (0, 0, 255), 1)
    # cv2.imwrite("contour.png" , frame_draw)

    return contours, frame_draw


def get_lines(frame, frame_draw=None, num_lines=10, line_width=5):
    """
    Generates vertical lines across an image and counts white pixels on each line.

    :param frame: black and white image to count white pixels from
    :param frame_draw: optional image to draw the lines on (new image if none is given)
    :param num_lines: lines to draw
    :param line_width: width of each line
    :return: returns line positions array with start and end positions for each line, array with white pixel counts
             for every line, and image with lines drawn on it
    """
    if frame_draw is None:
        frame_draw = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

    # segment width between lines (or at start and end)
    # 1 more segment than lines
    segment_width = frame.shape[1] // (num_lines + 1)

    white_pixel_counts = []
    line_positions = []
    for line in range(1, num_lines + 1):
        # calculate start and end columns
        start_col = line * segment_width
        end_col = start_col + line_width
        line_positions.append([start_col, end_col])

        # count white pixels and add them to counts for lines
        white_pixel_count = np.count_nonzero(frame[:, start_col:end_col] == 255)
        white_pixel_counts.append(white_pixel_count)

        # mark the line with green color
        frame_draw[:, start_col:end_col] = (0, 255, 0)

    # cv2.imwrite("line.png" , frame_draw)

    return line_positions, white_pixel_counts, frame_draw


def object_handler(frame, frame_color, frame_draw=None,
                   contour_threshold_max=100000, contour_threshold_min=4, max_num_contours=1,
                   ignore_centroids=None, ignore_centroids_distance=200,
                   contours=None, lines=None, line_positions=None, ignore_lines=True, num_lines=10):
    """
    Processes features to extract:
     - Selected contour centroids, median color, area covered
     - Median color of contours touching vertical lines.
     - Area covered by all contours, number of total contours, area covered by contours in thresholds (useful),
       and number of useful contours.

    :param frame: black and white image for processing features
    :param frame_color: image with color to get contour color from
    :param frame_draw: optional image to draw processed features on (new image if none is given)
    :param contour_threshold_max: max size for selected max contour
    :param contour_threshold_min: min size for selected max contour
    :param max_num_contours: number of contours to select and return
    :param ignore_centroids: if true ignore contours with centroids near ignored centroids (near determined by
                             ignore_centroids_distance)
    :param ignore_centroids_distance: max distance for a centroid to be counted as near (used with ignore_centroids)
    :param contours: contour objects already found (if None, find contours from black and white frame)
    :param lines: line white pixel counts array already found (if None, find counts from black and white frame if
                  ignore_lines is True)
    :param line_positions: line positions array with start and end positions for each line (if None, find positions from
                           black and white frame if ignore_lines is True)
    :param ignore_lines: if True ignore line touching logic, else contours that touch lines are not selected
    :param num_lines: number of lines (used when creating the line array when None are given)
    :return: returns centroids, colors, max_contour_areas, line_contour_colors, total_contour_area, total_contour_count,
             total_contour_useful_area, total_contour_useful_count, frame_draw. The first 3 parameters are None if no
             max contour is selected
    """
    if frame_draw is None:
        frame_draw = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
    if contours is None:
        contours, _ = get_contours(frame, frame_draw)
    if (lines is None or line_positions is None) and ignore_lines:
        line_positions, lines, _ = get_lines(frame, frame_draw, num_lines)

    # find max contour using the area of white pixels in it
    max_contours = []
    max_contour_areas = []
    max_contour_area = 0

    # total values
    total_contour_area = 0
    total_contour_count = len(contours)
    total_contour_useful_area = 0
    total_contour_useful_count = 0

    touches_line_colors = []
    for line in range(num_lines):
        touches_line_colors.append([])
    for contour in contours:
        contour_area = cv2.contourArea(contour)

        # get totals
        if contour_threshold_max >= contour_area >= contour_threshold_min:
            total_contour_useful_count += 1
            total_contour_useful_area += contour_area
        total_contour_area += contour_area

        # check line touching
        touches_line = False
        # only contours bigger than minimum threshold are considered
        if contour_area >= contour_threshold_min:
            touch_indexes = check_contour_line_overlap(contour, line_positions, frame, frame_draw)
            # if contour touches line (or lines)
            if len(touch_indexes) != 0:
                touches_line = True
                color = get_contour_color(contour, frame_color)
                for idx in touch_indexes:
                    touches_line_colors[idx].append(color)

        # if new max below threshold
        if contour_threshold_max >= contour_area > max_contour_area and contour_area >= contour_threshold_min:
            # if not ignoring contours that touch lines or if not touching line when ignoring
            if (not touches_line and ignore_lines) or not ignore_lines:
                # if parameter is selected, ignore contours with centroids near ignored centroids
                if ignore_centroids is not None:
                    # check if it's close to centroids in ignored centroids
                    moments = cv2.moments(contour)
                    centroid_x = int(moments["m10"] / moments["m00"])
                    centroid_y = int(moments["m01"] / moments["m00"])

                    ignore_centroid = False
                    for centroid in ignore_centroids:
                        # centroid is nearby, ignore contour
                        distance = np.sqrt((centroid_x - centroid[0]) ** 2 + (centroid_y - centroid[1]) ** 2)
                        if distance <= ignore_centroids_distance:
                            ignore_centroid = True
                            break

                    # max contour only if its centroid is not nearby an ignored centroid
                    if not ignore_centroid:
                        max_contour_area = contour_area
                        max_contour_areas.append(max_contour_area)
                        max_contours.append(contour)
                        if len(max_contours) > max_num_contours:
                            max_contours.pop(0)
                            max_contour_areas.pop(0)
                else:
                    max_contour_area = contour_area
                    max_contour_areas.append(max_contour_area)
                    max_contours.append(contour)
                    if len(max_contours) > max_num_contours:
                        max_contours.pop(0)
                        max_contour_areas.pop(0)

    # get medians of contour colors that touch each line
    line_contour_colors = []
    for line in range(num_lines):
        contour_colors = np.array(touches_line_colors[line])
        if contour_colors.size > 0:
            line_contour_colors.append(np.median(np.array(touches_line_colors[line]), axis=0).astype(int))
        else:
            line_contour_colors.append(np.array((-1, -1, -1)))

    if len(max_contour_areas) != 0:
        centroids = []
        colors = []
        for contour in max_contours:
            # dominant contour centroid
            moments = cv2.moments(contour)
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])

            # mark the center with a red circle
            frame_draw = cv2.circle(frame_draw, (centroid_x, centroid_y), 10, (0, 0, 255), -1)
            # cv2.imwrite("centroid.png" , frame_draw)
            centroids.append([centroid_x, centroid_y])

            # dominant contour color
            colors.append(get_contour_color(contour, frame_color))

        return centroids, colors, max_contour_areas, line_contour_colors, \
            total_contour_area, total_contour_count, total_contour_useful_area, total_contour_useful_count, frame_draw
    else:
        return None, None, None, line_contour_colors, \
            total_contour_area, total_contour_count, total_contour_useful_area, total_contour_useful_count, frame_draw


def process_video_frames(video_name, frame_start=0, frame_end=1000000, frame_interval=30,
                         output_plain_frames=False, output_processed_frames=False,
                         output_processed_video=False, output_processed_video_fps=1,
                         black_and_white_threshold=26,
                         contour_threshold_max=100000, contour_threshold_min=4,
                         ignore_centroids_max=4, ignore_centroids_distance=200):
    """
    Processes video frames by keeping only frames according to an interval. Processed results can be output to video.

    :param video_name: video name from videos directory
    :param frame_start: frame to start processing (to skip initial frames)
    :param frame_end: frame to stop processing (to skip ending frames)
    :param frame_interval: frames to skip (default 30 to keep 1 frame per second of a 30fps video)
    :param output_plain_frames: outputs frame images without processing to frames/plain directory if True
    :param output_processed_frames: outputs frame images with processing to frames/processed directory if True
    :param output_processed_video: outputs video created from processed frames to processed_videos directory if True
    :param output_processed_video_fps: frames per second for output video, default 1 (used with frame_interval 30)
    :param black_and_white_threshold: max value for black values, rest is white (used for processing)
    :param contour_threshold_max: max size for selected max contour (used for processing)
    :param contour_threshold_min: min size for selected max contour (used for processing)
    :param ignore_centroids_max: max previous centroids to be considered to be ignored (used for processing)
    :param ignore_centroids_distance: max distance for a centroid to be counted as near (used for processing)
    """
    # open video object
    video = cv2.VideoCapture(os.path.join("videos", video_name))
    if not video.isOpened():
        return

    # plain frames, no processing
    output_path_plain = os.path.join("frames", video_name.replace(".mp4", ""), "plain")
    if output_plain_frames:
        if os.path.exists(output_path_plain):
            shutil.rmtree(output_path_plain)
        os.makedirs(output_path_plain)

    # processed frames
    output_path_processed = os.path.join("frames", video_name.replace(".mp4", ""), "processed")
    if output_processed_frames:
        if os.path.exists(output_path_processed):
            shutil.rmtree(output_path_processed)
        os.makedirs(output_path_processed)

    # processed video
    output_path_processed_video = os.path.join("processed_videos", video_name.replace(".mp4", ""))
    if output_processed_video:
        if os.path.exists(output_path_processed_video):
            shutil.rmtree(output_path_processed_video)
        os.makedirs(output_path_processed_video)
    output_processed_video_frames = []  # output video frames

    frame_count = 0  # actual frame counter
    used_frame_count = 0  # used frame counter

    # ignoring previous nearby centroids
    if ignore_centroids_max != 0:
        ignore_centroids = []
    else:
        ignore_centroids = None

    while True:
        flag, frame_color = video.read()  # read frame from the video

        # stop if end of video is reached
        if not flag:
            break

        # check if frame will be used according to interval
        if frame_count % frame_interval == 0 and frame_start <= frame_count <= frame_end:
            used_frame_count += 1  # increment used frame counter

            # save the plain frame as an image
            if output_plain_frames:
                cv2.imwrite(os.path.join(output_path_plain, str(used_frame_count) + ".png"), frame_color)

            # if producing the processed video with processed frames
            if output_processed_video:
                frame_bw = black_and_white_video_frame(frame_color, black_and_white_threshold=black_and_white_threshold)

                contours, _ = get_contours(frame_bw)

                line_positions, lines, frame_draw = get_lines(frame_bw)

                found_centroids, contour_colors, contour_areas, line_colors, \
                    total_contour_area, total_contour_count, \
                    total_contour_useful_area, total_contour_useful_count, \
                    frame_draw = object_handler(frame_bw, frame_color, frame_draw,
                                                contour_threshold_max=contour_threshold_max,
                                                contour_threshold_min=contour_threshold_min,
                                                ignore_centroids=ignore_centroids,
                                                ignore_centroids_distance=ignore_centroids_distance,
                                                contours=contours, lines=lines, line_positions=line_positions)

                # previous nearby centroids
                if ignore_centroids is not None and found_centroids is not None:
                    for centroid in found_centroids:
                        ignore_centroids.append(centroid)
                        if len(ignore_centroids) > ignore_centroids_max:
                            ignore_centroids.pop(0)  # remove the oldest centroid

                output_processed_video_frames.append(frame_draw)

                # save the processed frame as an image
                if output_processed_frames:
                    cv2.imwrite(os.path.join(output_path_processed, str(used_frame_count) + ".png"), frame_draw)

        frame_count += 1  # increment the frame counter

    # release video object and close windows
    video.release()
    cv2.destroyAllWindows()

    if output_processed_video:
        create_video(output_processed_video_frames, os.path.join(output_path_processed_video, "processed.mp4"),
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
    fps = 30
    videos = pd.read_csv("videos.csv")
    for idx, video in videos.iterrows():
        print("Processing video: " + video["Video Name"])
        if video["Video Name"] == "A_Flight_to_HCG_40.mp4":
            process_video_frames(video["Video Name"], video["Start Time"] * fps, video["End Time"] * fps,
                                 frame_interval=30,
                                 output_plain_frames=True, output_processed_frames=True,
                                 output_processed_video=True, output_processed_video_fps=1,
                                 black_and_white_threshold=26,
                                 contour_threshold_max=100000, contour_threshold_min=4,
                                 ignore_centroids_max=4, ignore_centroids_distance=200
                                 )
        elif video["Video Name"] == "Flight_to_AG_Carinae.mp4":
            process_video_frames(video["Video Name"], video["Start Time"] * fps, video["End Time"] * fps,
                                 frame_interval=30,
                                 output_plain_frames=True, output_processed_frames=True,
                                 output_processed_video=True, output_processed_video_fps=1,
                                 black_and_white_threshold=26,
                                 contour_threshold_max=100000, contour_threshold_min=4,
                                 ignore_centroids_max=4, ignore_centroids_distance=200
                                 )
        else:
            process_video_frames(video["Video Name"], video["Start Time"] * fps, video["End Time"] * fps,
                                 frame_interval=30,
                                 output_plain_frames=False, output_processed_frames=False,
                                 output_processed_video=False)
