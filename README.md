Face Morph Research
=====================


The objective of this research project is to investigate the "Magic Passport" Security breach

https://www.researchgate.net/publication/283473746_The_magic_passport


Overview:

The main function is main.py. main.py calls the other functions, found in the scripts folder.
For loops to iterate through each of the images, landmark files, etc are found inside of the functions. However, you can adjust
the ranges to process fewer images by changing numStart and numEnd in main.py.
By default, these output 50/50 morphs.

 1) landmark.py
    * finds 68 facial landmark points of original image pairs and outputs .txt files of coordinates to /landmarks/Impostors_Output
    * 68 rows of numbers, first column is x coordinates, second column is y coordinates
    function: landmark(numStart, numEnd, imageFolder, outputFolder)
        - numStart is # of the first pair (start at 0)
        - numEnd is # of the last pair (end at 1000)
        - imageFolder is folder original image pairs (2000 images total)
        - outputFolder is folder where landmark points are outputted

 2) faceMorph.py
    * morphs faces using 68 coordinates and outputs morphs to Impostors_M_Output in images folder
    * change the alpha value (0 < alpha < 1.0) in the faceMorph() function to adjust the % resemblance to each pair. For instance,
     use alpha = .5 to make a 50/50 morph and alpha = .7 to make a 70% a, 30% b morph
    * Reference:
        - original tutorial: https://www.learnopencv.com/face-morph-using-opencv-cpp-python/
        - original source code: https://github.com/spmallick/learnopencv/tree/master/FaceMorph
    function: faceMorph(numStart, numEnd, imageFolder, outputFolder, morphsFolder)
        - first 4 arguments reference same things as landmark.py
        - morphsFolder is folder of 1000 morph images (face only, black background)

 3) landmark_morph.py
    * finds 68 facial landmark points of morph and outputs .txt file of coordinates to Impostors__M_Output in landmarks folder
    function
    * very similar to landmark.py, but gets landmarks of the morph instead of pairs of images
    function: landmark(numStart, numEnd, morphsFolder, morphOutput_Folder)
        - for range numStart to numEnd, find landmark points of images in morphsFolder and output as txt files to morphOutput_Folder

 4) k_skin.py
    * step that replaces black mask background of morphs with dominant skin color using K-means clustering
    * Notes:
        * takes the longest time to run
        * k_skin program makes 5 clusters, orders them in descending frequency, and chooses the 2nd most frequenct color
    * Reference:
        - https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
    function: k_skin(numStart, numEnd, morphsFolder, kMorphsFolder)
        - k-means "skin background" morphs are outputted to kMorphsFolder 

 5) remove_sideburns.py
    * goes through landmark files in /landmarks/Morphs and adjusts the x and y coordinate positions by a specified amount, to try to remove "sideburns
    function: remove_sideburns(numStart, numEnd, outputFolder, newOutputFolder)
        - takes txt files from outputFolder and outputs adjusted points to newOutputFoldder

 6) laplacian_swap.py
    * swaps morph face mask onto original face, along with laplacian blending
    * Reference:
        - original tutorial: https://www.learnopencv.com/face-swap-using-opencv-c-python/
        - original face swap code: https://github.com/spmallick/learnopencv/tree/master/FaceSwap
        - laplacian blending code: https://compvisionlab.wordpress.com/2013/05/13/image-blending-using-pyramid/ 
    function: laplacian_swap(numStart, numEnd, kMorphsFolder, imageFolder, morphOutputFolder, newOutputFolder, finalFolder)
        - final result goes to finalFolder
        - uses adjusted landmarks (via remove_sideburns.py) found in newOutputFolder

Notes: 
- images folder contains image files, including original pairs and morphs, k morphs, and final swapped morphs
- landmarks folder contains dlib landmark points in text files (68 lines of x, y coordinates)
    - Impostors_Output folder contains points of original a & b pairs
    - Impostors_M_Output folder contains points of morph
    - Impostors_Output_70_a is the for the 70% a pairs (morphs that look 70% like a), Impostors_Output_70_b is the 70% b pairs
    - Adjusted_Output folder contains files of Impostors_Output, but "adjusted" by remove_sideburns.py (i.e. coordinates a shifted inwards and downwards)
- old_versions folder contains previous iterations of program files, like the celebrity test morphs. Not necessary for the main program to run

## Installation

1) Install Anaconda Version 5.2.0
    * conda update
    * conda install anaconda=5.2

2) Set up Anaconda environment with Python 3, named my_env
    * conda create --name my_env python=3

3) Activate new environment
    * conda activate my_env

4) Install numpy
    * conda install --n my_env python=3 numpy

5) Install OpenCV 3.1.0
    * conda install -c menpo opencv3 

6) Install matplotlib 3.0.3
    * conda install --n my_env matplotlib

7) Install scipy 
    * conda install --n my_env scipy

8) Install dlib
    * conda install --n my_env -c menpo dlib

* please let me know if you have any issues with installation! I'm not 100% about some of these commands, but tried to remember the best I could
Also, I may have used pip for some of these programs

## Run

1) Activate Anaconda environment
    * source activate my_env

2) Run main program
    * python main.py

3) Look at Pictures
    * the final pictures of morphed & swapped faces are found in the Swapped_Morphs_Lap folder in the images folder
    * the 70a/30b pictures are found in Swapped_Morphs_Lap_70_a
    * the 30a/70b pictures are found in Swapped_Morphs_Lap_70_b
    * intermediate or alternate pictures are found in the remaining folders of the images folder

Notes:
* At this point, there are 3 versions of the faceSwap stage. 
1) faceswap_lap.py, which is called in main using the function laplacian_swap()
    * Newest and most successful method
    * uses dominant skin color + laplacian blending + remove_sideburns.py
2) faceSwap2.py, the oldest version, found in old_versions
3) using faceSwap2.py in tandem with k_skin

* There are 5 invalid out of bounds image pairs (29, 352, 662, 677, 841), which are handled by try/except statements. They shouldn't pose any issue, but it means there are
a total of 995 valid pairs instead of 1000
* To save time, comment out the functions you do not need to run. Steps 1 - 4 have already been run, and the results
  are found in the landmark and images folders
* Feel free to make new folders, and adjust the names in main.py accordingly. I would recommend making new folders for testing to avoid overwriting existing files

Resources:

https://www.learnopencv.com/face-swap-using-opencv-c-python/
https://www.learnopencv.com/face-morph-using-opencv-cpp-python/
