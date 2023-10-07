import cv2
import os
import numpy as np
import pandas as pd
import shutil
from keras.models import load_model


def black_and_white_video_frame(frame_color, black_and_white_threshold=26,
                                dilation_iterations=0, dilation_size=5,
                                erosion_iterations=0, erosion_size=5):
    """
    Returns black and white image from colored image with a certain threshold for values.

    :param frame_color: colored image
    :param black_and_white_threshold: max value for black values, rest is white
    :param dilation_iterations: iterations to run dilation on image, 0 to not run
    :param dilation_size: size of dilation matrix
    :param erosion_iterations: iterations to run erosion on image, 0 to not run
    :param erosion_size: size of erosion matrix
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

    # dilate image
    if dilation_iterations > 0:
        dilation_kernel = np.ones((dilation_size, dilation_size), np.uint8)

        frame_bw = cv2.dilate(frame_bw, dilation_kernel, iterations=dilation_iterations)

    # erode image
    if erosion_iterations > 0:
        erosion_kernel = np.ones((erosion_size, erosion_size), np.uint8)

        frame_bw = cv2.erode(frame_bw, erosion_kernel, iterations=erosion_iterations)

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
        frame_draw = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

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


def check_contour_is_big_celestial(contour, classification_model, frame, frame_model, frame_color, frame_draw=None):
    """
    Checks if contour has overlap with binary classification model's mask prediction for the frame. If there is overlap
    then the contour is a big celestial object.

    :param contour: contour object
    :param classification_model: binary classification model to predict a binary mask for frame
    :param frame: black and white image to get contour area from
    :param frame_model: binary mask from binary classification model
    :param frame_color: image with color for model to predict a binary mask
    :param frame_draw: optional image to draw the centroid of the contour on if it's classified as a big celestial
                       (new image if none is given)
    :return: returns True if contour is a big celestial, otherwise False
    """
    if frame_draw is None:
        frame_draw = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # create an empty mask
    contour_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # use the mask to extract the region within the contour
    contour_roi = cv2.bitwise_and(frame, frame, mask=contour_mask)

    # bitwise end to check overlap with model's binary classification mask
    overlap_mask = cv2.bitwise_and(contour_roi, frame_model)
    overlap_exists = np.any(overlap_mask != 0)
    if overlap_exists:
        # draw contour centroid with yellow color
        moments = cv2.moments(contour)
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
        frame_draw = cv2.circle(frame_draw, (centroid_x, centroid_y), 10, (0, 255, 255), -1)

        return True

    return False


def get_contours(frame, frame_draw=None):
    """
    Generates contours on an image.

    :param frame: black and white image to get contours out of
    :param frame_draw: optional image to draw the contours on (new image if none is given)
    :return: returns contour objects and image with contours drawn on it
    """
    if frame_draw is None:
        frame_draw = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

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
        frame_draw = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

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
                   contours=None, lines=None, line_positions=None, ignore_lines=True, num_lines=10,
                   classification_model=None):
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
    :param classification_model: classification model used for big celestial object detection, jf None ignore big
                                 celestials
    :return: returns centroids, colors, max_contour_areas, line_contour_colors, total_contour_area, total_contour_count,
             total_contour_useful_area, total_contour_useful_count, frame_draw. The first 3 parameters are None if no
             max contour is selected
    """
    if frame_draw is None:
        frame_draw = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
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

    # big celestials
    big_celestial_colors = []
    big_celestial_area = 0
    # get binary classifications mask from model
    if classification_model is not None:
        frame_model = classification_model.predict(np.expand_dims(frame_color, axis=0))[0]
        frame_model = clear_classification(frame_model)  # clear classification

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

        # big celestial classification
        is_big_celestial = False
        if classification_model is not None:
            is_big_celestial = check_contour_is_big_celestial(contour, classification_model,
                                                              frame, frame_model, frame_color, frame_draw)
            if is_big_celestial:
                big_celestial_colors.append(get_contour_color(contour, frame_color))
                big_celestial_area += contour_area

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

        # if new max below max threshold and above min threshold and is not a big celestial object
        if contour_threshold_max >= contour_area > max_contour_area and contour_area >= contour_threshold_min and \
                not is_big_celestial:
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

    # get medians of contour colors for big celestials
    if len(big_celestial_colors) == 0:
        big_celestial_color = np.median(np.array(big_celestial_colors), axis=0).astype(int)
    else:
        big_celestial_color = np.array((-1, -1, -1))

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

        return centroids, colors, max_contour_areas, line_contour_colors,\
            big_celestial_color, big_celestial_area, \
            total_contour_area, total_contour_count, total_contour_useful_area, total_contour_useful_count, frame_draw
    else:
        return None, None, None, line_contour_colors, \
            big_celestial_color, big_celestial_area, \
            total_contour_area, total_contour_count, total_contour_useful_area, total_contour_useful_count, frame_draw


def clear_classification(frame_model):
    """
    Clears misclassifications in model prediction.

    :param frame_model: frame produced by model
    :return: returns cleared frame
    """
    frame_model[frame_model >= 0.99] = 255
    frame_model[frame_model < 0.99] = 0

    erosion_kernel_initial = np.ones((8, 8), np.uint8)
    erosion_kernel_final = np.array([[1, 1, 1],
                                     [1, 0, 1],
                                     [1, 1, 1]], dtype=np.uint8)

    dilation_kernel = np.ones((4, 4), np.uint8)

    eroded_initial = cv2.erode(frame_model.copy(), erosion_kernel_initial, iterations=3)
    dilated = cv2.dilate(eroded_initial, dilation_kernel, iterations=2)
    eroded_final = cv2.erode(dilated, erosion_kernel_final, iterations=5)

    return eroded_final.astype(np.uint8)


def get_lines_dictionary(line_positions, lines, line_colors):
    """
    Creates dictionary of extracted line features and information.

    :param line_positions: array with start and end x positions of each line
    :param lines: white pixel counts for every line array
    :param line_colors: median color of contours touching a line for every line array
    :return: returns dictionary of white pixel counts, line positions, and median colors
    """
    line_dict = {}

    for idx, line in enumerate(lines):
        line_choice = "_line" + str(idx)
        line_dict["whitecount" + line_choice] = line
        line_dict["start" + line_choice] = line_positions[idx][0]
        line_dict["end" + line_choice] = line_positions[idx][1]
        line_dict["median_red_contours_touching" + line_choice] = line_colors[idx][2]
        line_dict["median_green_contours_touching" + line_choice] = line_colors[idx][1]
        line_dict["median_blue_contours_touching" + line_choice] = line_colors[idx][0]

    return line_dict


def get_contours_dictionary(contour_centroids, contour_colors, contour_areas,
                            total_contour_area, total_contour_count,
                            total_contour_useful_area, total_contour_useful_count):
    """
    Creates dictionary of extracted contour features and information.

    :param contour_centroids: array of selected contour centroids, if None no contour is selected
    :param contour_colors: array of selected contour median colors, if None no contour is selected
    :param contour_areas: array of selected contour areas, if None no contour is selected
    :param total_contour_area: total contour area in frame
    :param total_contour_count: total number of contours in frame
    :param total_contour_useful_area: total useful contour area in frame
    :param total_contour_useful_count: total number of useful contours in frame
    :return: returns dictionary of contour centroids, colors, area, total area and useful area, and total contour and
             useful contour count
    """
    contour_dict = {}

    # no contour selected
    if contour_centroids is None:
        contour_choice = "_contour" + str(0)
        contour_dict["x" + contour_choice] = -1
        contour_dict["y" + contour_choice] = -1
        contour_dict["median_red" + contour_choice] = -1
        contour_dict["median_green" + contour_choice] = -1
        contour_dict["median_blue" + contour_choice] = -1
        contour_dict["area" + contour_choice] = -1

    # for each saved contour selected
    for idx, contour in enumerate(contour_centroids):
        contour_choice = "_contour" + str(idx)
        contour_dict["x" + contour_choice] = contour[0]
        contour_dict["y" + contour_choice] = contour[1]
        contour_dict["median_red" + contour_choice] = contour_colors[idx][2]
        contour_dict["median_green" + contour_choice] = contour_colors[idx][1]
        contour_dict["median_blue" + contour_choice] = contour_colors[idx][0]
        contour_dict["area" + contour_choice] = contour_areas[idx]

    # add totals
    contour_dict["total_area"] = total_contour_area
    contour_dict["total_contour_count"] = total_contour_count
    contour_dict["total_contour_useful_area"] = total_contour_useful_area
    contour_dict["total_contour_useful_count"] = total_contour_useful_count

    return contour_dict


def get_big_celestial_dictionary(big_celestial_color, big_celestial_area):
    """
    Creates dictionary of extracted big celestial features and information.

    :param big_celestial_color: median color of celestial object
    :param big_celestial_area: total area of celestial object
    :return: returns dictionary of big celestial median color and total area
    """
    return {
        "big_celestial_total_area": big_celestial_area,
        "big_celestial_median_red": big_celestial_color[2],
        "big_celestial_median_green": big_celestial_color[1],
        "big_celestial_median_blue": big_celestial_color[0]
    }

def process_video_frames(video_name, frame_start=0, frame_end=1000000, frame_interval=30,
                         output_plain_frames=False, output_processed_frames=False,
                         output_processed_video=False, output_processed_video_fps=1,
                         black_and_white_threshold=26,
                         dilation_iterations=0, dilation_size=5,
                         erosion_iterations=0, erosion_size=5,
                         contour_threshold_max=100000, contour_threshold_min=4,
                         ignore_centroids_max=4, ignore_centroids_distance=200,
                         classification_model=None):
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
    :param dilation_iterations: iterations to run dilation on image, 0 to not run (used for processing)
    :param dilation_size: size of dilation matrix (used for processing)
    :param erosion_iterations: iterations to run erosion on image, 0 to not run (used for processing)
    :param erosion_size: size of erosion matrix (used for processing)
    :param contour_threshold_max: max size for selected max contour (used for processing)
    :param contour_threshold_min: min size for selected max contour (used for processing)
    :param ignore_centroids_max: max previous centroids to be considered to be ignored (used for processing)
    :param ignore_centroids_distance: max distance for a centroid to be counted as near (used for processing)
    :param classification_model: classification model used for big celestial object detection (used for processing)
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

    # initialize data dataframes
    line_data = pd.DataFrame()
    contour_data = pd.DataFrame()
    big_celestial_data = pd.DataFrame()

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
                frame_bw = black_and_white_video_frame(frame_color,
                                                       black_and_white_threshold=black_and_white_threshold,
                                                       dilation_iterations=dilation_iterations,
                                                       dilation_size=dilation_size,
                                                       erosion_iterations=erosion_iterations,
                                                       erosion_size=erosion_size)

                contours, _ = get_contours(frame_bw)

                line_positions, lines, frame_draw = get_lines(frame_bw)

                found_centroids, contour_colors, contour_areas, line_colors,\
                    big_celestial_color, big_celestial_area, \
                    total_contour_area, total_contour_count, \
                    total_contour_useful_area, total_contour_useful_count, \
                    frame_draw = object_handler(frame_bw, frame_color, frame_draw,
                                                contour_threshold_max=contour_threshold_max,
                                                contour_threshold_min=contour_threshold_min,
                                                ignore_centroids=ignore_centroids,
                                                ignore_centroids_distance=ignore_centroids_distance,
                                                contours=contours, lines=lines, line_positions=line_positions,
                                                classification_model=classification_model)

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

                # save data for lines in processed frame
                line_dict = get_lines_dictionary(line_positions, lines, line_colors)
                line_data = pd.concat([line_data, pd.DataFrame([line_dict])], ignore_index=True)

                # save data for contours in processed frame
                contour_dict = get_contours_dictionary(found_centroids, contour_colors, contour_areas,
                                                       total_contour_area, total_contour_count,
                                                       total_contour_useful_area, total_contour_useful_count)
                contour_data = pd.concat([contour_data, pd.DataFrame([contour_dict])], ignore_index=True)

                # save data for big celestial objects in processed frame
                big_celestial_dict = get_big_celestial_dictionary(big_celestial_color, big_celestial_area)
                big_celestial_data = pd.concat([big_celestial_data, pd.DataFrame([big_celestial_dict])],
                                               ignore_index=True)

        frame_count += 1  # increment the frame counter

    # release video object and close windows
    video.release()
    cv2.destroyAllWindows()

    if output_processed_video:
        create_video(output_processed_video_frames, os.path.join(output_path_processed_video, "processed.mp4"),
                     fps=output_processed_video_fps)

        output_path_data = os.path.join("data", video_name.replace(".mp4", ""))
        if os.path.exists(output_path_data):
            shutil.rmtree(output_path_data)
        os.makedirs(output_path_data)

        # save data for processed frames of video
        line_data.to_csv(os.path.join(output_path_data, "lines.csv"), index=False)
        contour_data.to_csv(os.path.join(output_path_data, "contours.csv"), index=False)
        big_celestial_data.to_csv(os.path.join(output_path_data, "big_celestials.csv"), index=False)


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
    model = load_model("classifier.h5")

    fps = 30
    videos = pd.read_csv("videos.csv")
    for idx, video in videos.iterrows():
        print("Processing video: " + video["Video Name"])

        if video["Video Name"] == "Flight_to_AG_Carinae.mp4":
            process_video_frames(video["Video Name"], video["Start Time"] * fps, video["End Time"] * fps,
                                 frame_interval=30,
                                 output_plain_frames=True, output_processed_frames=True,
                                 output_processed_video=True, output_processed_video_fps=1,
                                 black_and_white_threshold=26,
                                 contour_threshold_max=100000, contour_threshold_min=4,
                                 ignore_centroids_max=4, ignore_centroids_distance=200,
                                 classification_model=model
                                 )
        """
        elif video["Video Name"] == "A_Flight_to_HCG_40.mp4":
            process_video_frames(video["Video Name"], video["Start Time"] * fps, video["End Time"] * fps,
                                 frame_interval=30,
                                 output_plain_frames=False, output_processed_frames=True,
                                 output_processed_video=True, output_processed_video_fps=1,
                                 black_and_white_threshold=55,
                                 dilation_iterations=2, dilation_size=5,
                                 erosion_iterations=0, erosion_size=5,
                                 contour_threshold_max=100000, contour_threshold_min=4,
                                 ignore_centroids_max=3, ignore_centroids_distance=200,
                                 classification_model=model
                                 )
        else:
            process_video_frames(video["Video Name"], video["Start Time"] * fps, video["End Time"] * fps,
                                 frame_interval=30,
                                 output_plain_frames=False, output_processed_frames=False,
                                 output_processed_video=False)
        """