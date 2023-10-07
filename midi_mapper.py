import pandas as pd
import librosa
from midiutil import MIDIFile
import numpy as np
import os
import shutil


def import_data(csv_file_name, csv_file_path):
    filename = csv_file_name  # name of csv data file
    path = csv_file_path

    df = pd.read_csv(path + '/' + filename)  # load data as a pandas dataframe
    return df


def notes_to_midi(notes_lst):
    line_midi_code_notes = []
    for note in notes_lst: line_midi_code_notes.append(librosa.note_to_midi(note))
    return line_midi_code_notes


def line_vel_midi(df, n_lines, threshold):
    line_vel_high = pd.DataFrame()
    line_vel_low = pd.DataFrame()

    for i in range(n_lines):
        line_vel_high['line' + str(i)] = df['whitecount_line' + str(i)]
        line_vel_low['line' + str(i)] = df['whitecount_line' + str(i)]

        line_vel_high.loc[line_vel_high['line' + str(i)] < threshold, 'line' + str(i)] = 0
        line_vel_low.loc[line_vel_low['line' + str(i)] > threshold, 'line' + str(i)] = 0
        line_vel_low.loc[line_vel_low['line' + str(i)] < 30, 'line' + str(i)] = 0

    return line_vel_high, line_vel_low


def norm_scale(value, min_value, max_value, min_result, max_result, inverted=False):
    if inverted:
        normalized_array = 1 - (value - min_value) / (max_value - min_value)
        scaled_array = normalized_array * (max_result - min_result)
    else:
        normalized_array = (value - min_value) / (max_value - min_value)
        scaled_array = normalized_array * (max_result - min_result)
    return scaled_array


def create_lines_midi_file(df, n_lines, threshold, notes_lst, tempo):
    line_midi_code_notes = notes_to_midi(notes_lst)
    # line_midi_code_notes = [57, 58, 60, 62, 64, 65, 67, 69, 76, 79]
    high_midi_file = MIDIFile(1)
    high_midi_file.addTempo(track=0, time=0, tempo=tempo)
    low_midi_file = MIDIFile(1)
    low_midi_file.addTempo(track=0, time=0, tempo=tempo)
    line_vel_high, line_vel_low = line_vel_midi(df, n_lines, threshold)
    line_vel_high, line_vel_low = norm_scale(line_vel_high, 0, np.amax(line_vel_high.values), 0, 127,
                                             inverted=True), norm_scale(line_vel_low, 0, np.amax(line_vel_low.values),
                                                                        0, 127)
    line_vel_high.replace(to_replace=127.0, value=0)

    for i in range(n_lines):
        line_vel_high.loc[line_vel_high['line' + str(i)] > 126, 'line' + str(i)] = 0

    #print(line_vel_high)
    for index in line_vel_high.index:
        for line_num in range(n_lines):
            high_midi_file.addNote(track=0, channel=0, pitch=line_midi_code_notes[line_num], duration=1,
                                   volume=round(line_vel_high['line' + str(line_num)][index]), time=index)

    for index in line_vel_low.index:
        for line_num in range(n_lines):
            low_midi_file.addNote(track=0, channel=0, pitch=line_midi_code_notes[line_num], duration=1,
                                  volume=round(line_vel_low['line' + str(line_num)][index]), time=index)

    with open(os.path.join("midi", 'Low_thresh_midi' + '.mid'), "wb") as f:
        low_midi_file.writeFile(f)

    with open(os.path.join("midi", 'High_thresh_midi' + '.mid'), "wb") as f:
        high_midi_file.writeFile(f)


def star_vel(df):
    star_vel = df['area_contour0']
    norm_scale(star_vel, min(star_vel), max(star_vel), 35, 127)
    return star_vel


def create_star_midi_files(df):
    star_midi_file = MIDIFile(1)
    star_midi_file.addTempo(track=0, time=0, tempo=1)
    star_vel_lst = star_vel(df)

    chord_notes_lst = scale_notes([2, 3, 4])
    star_notes_lst = scale_notes([4, 5])
    root = 69

    for index, star in df.iterrows():
        rgb_lst = [star['median_red_contour0'], star['median_green_contour0'], star['median_blue_contour0']]
        if rgb_lst.index(max(rgb_lst)) == 0: melody_red(star_midi_file, star_notes_lst, 69, d=0.25, t=index,
                                                        velocity=100)  # velocity=star_vel_lst[index])
        if rgb_lst.index(max(rgb_lst)) == 1: melody_green(star_midi_file, star_notes_lst, 69, d=0.25, t=index,
                                                          velocity=100)  # velocity=star_vel_lst[index])
        if rgb_lst.index(max(rgb_lst)) == 2: melody_blue(star_midi_file, star_notes_lst, 69, d=0.25, t=index,
                                                         velocity=100)  # velocity=star_vel_lst[index])

    with open(os.path.join("midi", 'stars_midi' + '.mid'), "wb") as f:
        star_midi_file.writeFile(f)


def nebula_midi_file(chord_notes_list, big_celestial_volumes, d, proggression=False, time_idx=30):
    if not proggression:
        time_index = time_idx
        my_midi_file = MIDIFile(1)
        my_midi_file.addTimeSignature(numerator=4, denominator=4, clocks_per_tick=24, notes_per_quarter=8, time=0,
                                      track=0)
        root = 45
        my_midi_file.addTempo(track=0, time=0, tempo=80)
        if time_index % 4 == 0:
            for t in range(0, time_index + 1, 4):
                chords(my_midi_file, chord_notes_list, root, d, t)
        elif time_index % 3 == 0:
            for t in range(0, time_index + 1, 3):
                chords(my_midi_file, chord_notes_list, root, d, t)
        else:
            for t in range(0, time_index + 1, 2):
                chords(my_midi_file, chord_notes_list, root, d, t)
    else:
        time_index = time_idx
        my_midi_file = MIDIFile(1)
        root = 45
        first_index = chord_notes_list.index(root)
        my_midi_file.addTempo(track=0, time=0, tempo=60)
        if time_index % 5 == 0:
            for t in range(0, time_index, 5):
                velocity = int(big_celestial_volumes.iloc[t] * 100)
                chords(my_midi_file, chord_notes_list, root, 1, t, velocity=velocity)
                chords(my_midi_file, chord_notes_list, chord_notes_list[first_index + 2], d, t + 1, velocity=velocity)  # b3
                chords(my_midi_file, chord_notes_list, chord_notes_list[first_index + 1], d, t + 2, velocity=velocity)  # b2
                chords(my_midi_file, chord_notes_list, chord_notes_list[first_index + 5], d, t + 3, velocity=velocity)  # 6
                chords(my_midi_file, chord_notes_list, chord_notes_list[first_index + 6], d, t + 4, velocity=velocity)  # 7
        if time_index % 4 == 0:
            for t in range(0, time_index, 4):
                velocity = int(big_celestial_volumes.iloc[t] * 100)
                chords(my_midi_file, chord_notes_list, root, 1, t, velocity=velocity)
                chords(my_midi_file, chord_notes_list, chord_notes_list[first_index + 3], d, t + 1, velocity=velocity)  # 4
                chords(my_midi_file, chord_notes_list, chord_notes_list[first_index + 2], d, t + 2, velocity=velocity)  # b3
                chords(my_midi_file, chord_notes_list, chord_notes_list[first_index + 6], d, t + 3, velocity=velocity)  # 7
        elif time_index % 3 == 0:
            for t in range(0, time_index, 3):
                velocity = int(big_celestial_volumes.iloc[t] * 100)
                chords(my_midi_file, chord_notes_list, root, 1, t, velocity=velocity)
                chords(my_midi_file, chord_notes_list, chord_notes_list[first_index + 5], d, t + 1, velocity=velocity)  # b6
                chords(my_midi_file, chord_notes_list, chord_notes_list[first_index + 4], d, t + 2, velocity=velocity)  # 4
        else:
            for t in range(0, time_index, 2):
                velocity = int(big_celestial_volumes.iloc[t] * 100)
                chords(my_midi_file, chord_notes_list, root, 1, t, velocity=velocity)
                chords(my_midi_file, chord_notes_list, chord_notes_list[first_index + 1], d, t + 1, velocity=velocity)  # b2

    with open(os.path.join("midi", 'nebula_midi' + '.mid'), "wb") as f:
        my_midi_file.writeFile(f)
    #return my_midi_file



def big_object_vel(df):
    big_object_vel = df['big_celestial_total_area']
    big_object_vel = norm_scale(big_object_vel, min(big_object_vel), max(big_object_vel), 0, 1)

    return big_object_vel


def bigCelestialObject_midi(df):
    star_midi_file = MIDIFile(1)
    star_midi_file.addTempo(track=0, time=0, tempo=1)
    big_object_vel(df)


def background_music(my_midi_file, notes_list, line_vel, root, d):
    time_index = len(line_vel.index)
    note_index = notes_list.index(root)
    dt = d / 8
    for t in range(0, time_index, d):
        my_midi_file.addTempo(track=0, time=0, tempo=80)
        my_midi_file.addNote(track=0, channel=0, pitch=root, duration=d, volume=50, time=t)
        my_midi_file.addNote(track=0, channel=0, pitch=notes_list[note_index + 1], duration=d, volume=50,
                             time=t + 1 * dt)
        my_midi_file.addNote(track=0, channel=0, pitch=notes_list[note_index + 3], duration=d, volume=50,
                             time=t + 2 * dt)
        my_midi_file.addNote(track=0, channel=0, pitch=notes_list[note_index + 6], duration=d, volume=50,
                             time=t + 3 * dt)
        my_midi_file.addNote(track=0, channel=0, pitch=notes_list[note_index + 4], duration=d, volume=50,
                             time=t + 4 * dt)
        my_midi_file.addNote(track=0, channel=0, pitch=notes_list[note_index + 3], duration=d, volume=50,
                             time=t + 5 * dt)
        my_midi_file.addNote(track=0, channel=0, pitch=notes_list[note_index + 1], duration=d, volume=50,
                             time=t + 6 * dt)
        my_midi_file.addNote(track=0, channel=0, pitch=root, duration=d, volume=50, time=t + 7 * dt)
    with open(os.path.join("midi", "background_music.mid"), "wb") as midi_file:
        my_midi_file.writeFile(midi_file)


def scale_notes(octave_select):
    a_phrygian = []
    a_phrygian_octave_2 = [45, 46, 48, 50, 52, 53, 55]  # A2 to G2
    a_phrygian_octave_3 = [57, 58, 60, 62, 64, 65, 67]  # A3 to G3
    a_phrygian_octave_4 = [69, 70, 72, 74, 76, 77, 79]  # A4 to G4
    a_phrygian_octave_5 = [81, 82, 84, 86, 88, 89, 91]  # A5 to G5
    a_phrygian_octave_6 = [93, 94, 96, 98, 100, 101, 103]  # A6 to G6
    if 2 in octave_select:
        a_phrygian.extend(a_phrygian_octave_2)
    if 3 in octave_select:
        a_phrygian.extend(a_phrygian_octave_3)
    if 4 in octave_select:
        a_phrygian.extend(a_phrygian_octave_4)
    if 5 in octave_select:
        a_phrygian.extend(a_phrygian_octave_5)
    if 6 in octave_select:
        a_phrygian.extend(a_phrygian_octave_6)
    return a_phrygian


def chords(my_midi_file, scale_notes, root, d, t, velocity=100):
    index_root = scale_notes.index(root)
    #print(index_root)
    my_midi_file.addNote(track=0, channel=0, pitch=root, duration=d, volume=velocity, time=t)
    my_midi_file.addNote(track=0, channel=0, pitch=scale_notes[index_root + 2], duration=d, volume=velocity, time=t)
    my_midi_file.addNote(track=0, channel=0, pitch=scale_notes[index_root + 4], duration=d, volume=velocity, time=t)
    my_midi_file.addNote(track=0, channel=0, pitch=scale_notes[index_root + 6], duration=d, volume=velocity, time=t)


def melody_red(my_midi_file, star_notes, root, d, t, velocity):
    index_root = star_notes.index(root)
    my_midi_file.addNote(track=0, channel=0, pitch=root, duration=d, volume=velocity, time=t)
    my_midi_file.addNote(track=0, channel=0, pitch=star_notes[index_root + 1], duration=d, volume=velocity,
                         time=t + 1 / 4)  # b2
    my_midi_file.addNote(track=0, channel=0, pitch=star_notes[index_root + 4], duration=d, volume=velocity,
                         time=t + 1 / 2)  # 5
    my_midi_file.addNote(track=0, channel=0, pitch=star_notes[index_root + 5], duration=d, volume=velocity,
                         time=t + 3 / 4)  # 6


def melody_blue(my_midi_file, star_notes, root, d, t, velocity):
    index_root = star_notes.index(root)
    my_midi_file.addNote(track=0, channel=0, pitch=root, duration=d, volume=velocity, time=t)
    my_midi_file.addNote(track=0, channel=0, pitch=star_notes[index_root + 1], duration=d, volume=velocity,
                         time=t + 1 / 4)  # b2
    my_midi_file.addNote(track=0, channel=0, pitch=star_notes[index_root + 4], duration=d, volume=velocity,
                         time=t + 2 / 4)  # 5
    my_midi_file.addNote(track=0, channel=0, pitch=star_notes[index_root + 3], duration=d, volume=velocity,
                         time=t + 3 / 4)  # 4


def melody_green(my_midi_file, star_notes, root, d, t, velocity):
    index_root = star_notes.index(root)
    my_midi_file.addNote(track=0, channel=0, pitch=star_notes[index_root + 2], duration=d, volume=velocity, time=t)  # 3
    my_midi_file.addNote(track=0, channel=0, pitch=star_notes[index_root + 1], duration=d, volume=velocity,
                         time=t + 1 / 4)  # b2
    my_midi_file.addNote(track=0, channel=0, pitch=star_notes[index_root + 4], duration=d, volume=velocity,
                         time=t + 2 / 4)  # 5
    my_midi_file.addNote(track=0, channel=0, pitch=star_notes[index_root + 7], duration=d, volume=velocity,
                         time=t + 3 / 4)  # 1


if __name__ == "__main__":
    if os.path.exists("midi"):
        shutil.rmtree("midi")
    os.makedirs("midi")

    notes_list_str = ['G3', 'A3', 'B3', 'D4', 'E4', 'G4', 'A4', 'B4', 'D5', 'E5']
    notes_list_int = [45, 46, 48, 50, 52, 53, 55, 57, 58, 60, 62, 64, 65, 67]
    note_code = notes_to_midi(['A2', 'A#2', 'C3', 'D3', 'E3', 'F3', 'G3', 'A4'])

    # c lydian
    # C1, C2, G2, C3, E3,
    # G3, A3, B3, D4, E4, G4, A4, B4, D5, E5,
    # G5, A5, B5, D6, E6, F  # 6, G6, A6

    ## LINES
    df = import_data('lines.csv', 'data/Flight_to_AG_Carinae')
    create_lines_midi_file(df, n_lines=10, threshold=200, notes_lst=notes_list_str, tempo=1)

    ## STARS
    df = import_data('contours.csv', 'data/Flight_to_AG_Carinae')
    create_star_midi_files(df)

    ## NEBULA
    df = import_data('big_celestials.csv', 'data/Flight_to_AG_Carinae')
    df = big_object_vel(df)

    nebula_midi_file(notes_list_int, df, d=1, proggression=True)
