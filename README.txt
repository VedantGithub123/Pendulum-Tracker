NECESSARY PYTHON LIBRARIES: OpenCV, Numpy, Matplotlib, SciPy

HOW TO USE:
    1: Modify "mass-color-filter.txt" to the correct bounds for your mass
    2: Modify "reference-color-filter.txt" to the correct bounds for your reference block
        i:   The reference block is at the pivot point
    3: Add a video to folder"filter-testing-images" to test the filters. The image shown will be the first frame of the video
    4: Modify "frame-rate.txt" to the frame rate of the videos
    5: Modify "uncertainties.txt" to the correct type B uncertainties
    6: Add all videos to the "videos" folder
        i:   The video name should just be "length-initial angle-trial number-video"
        ii:  For example, if the initial angle is 25.78 degrees and the length is 10.5 and the trial is 12, the file name should be "10-5-25-78-12-video"
        iii: If the initial angle is a whole number like 25 degrees and the length is 10, use "10_0_25_0_12-video"
        iV:  There can be any number of decimal points
    7: Run "videos-to-position-data.py", you should now have text files in folder "position-data"
        i:   These new files will have the same name as the video, except replacing "video" with "position-data"
    8: Run "position-data-to-graphs.py", you should now have images in folder "position-graphs"
        i:   The images will have the same name as the video, except replacing "video" with "graph"
        ii:  These graphs will show all parameters and a line of best fit along with the period, q-factor, standard deviation,  
    9: You should also have images in folder "period-graphs"
        ii:  Another graph will be created which shows the period vs length
        iii: Another graph will be created which shows the period vs angle
        iV:  A 3d graph will be created which shows period vs length/angle
    10: You should also have images in folder "dampening-graphs"
        ii:  Another graph will be created which shows the dampening vs length
        iii: Another graph will be created which shows the dampening vs angle
        iV:  A 3d graph will be created which shows dampening vs length/angle
    11: You should also have images in folder "q-factor-graphs"
        ii:  Another graph will be created which shows the q-factor vs length
        iii: Another graph will be created which shows the q-factor vs angle
        iV:  A 3d graph will be created which shows q-factors vs length/angle


POSITION DATA FILE FORMAT:

Time1 Angle1
Time2 Angle2
...
TimeN AngleN