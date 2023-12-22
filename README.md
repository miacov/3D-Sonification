# 3D-Sonification
Sonification and Music Creation from 3D Fly-By Space Videos

## About
This project was created for NASA's Space Apps Challenge 2023.

For more information please refer to our team site:  
https://www.spaceappschallenge.org/2023/find-a-team/synastrix/?tab=project

Try out a demo of the sonification:
https://miacov.github.io/3D-Sonification/  
Sounds from different instruments can be played or muted using combinations of the  "Lows", "Highs", "Nebula", "Stars", and "Background (Extra)" buttons. The source of each sound can visualized on the source video by clicking on the the "Mapping Visualization" tab.

## Short Description
Processed 3D fly-by space videos and used a multi-factor sonification approach to generate MIDI files that are used for music production.
The produced music files can be found under `music` and are split into different tracks for every video.

### Sonification Approach to Produce MIDI Files
A vertical line grid is used on the videos and acts as notes played on a scale when objects in a frame meet it.
Contour detection was used to identify stars and use their size and color to play melodies with corresponding tension.
A binary-classification model was trained to detect big celestial bodies to assign chord progressions to them as bass.

![Processing](/frames/Flight_to_AG_Carinae/processed/2.png)
