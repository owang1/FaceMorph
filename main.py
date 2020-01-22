# main.py

from scripts.landmark import *
from scripts.faceMorph import *
from scripts.faceSwap import *
from scripts.k_skin import *
from scripts.landmark_morph import *
from scripts.laplacian_swap import *
from scripts.remove_sideburns import *
from faceSwap_lap import *

if __name__ == '__main__':
# Range of image pairs to process
    numStart = 33
    numEnd = 100
# Invalid Pairs: 29, 352, 662, 677, 841

# Step 1: Get facial landmarks of pairs
    imageFolder = "./images/Original_Pairs/"
    outputFolder = "./landmarks/Impostors_Output/"
    # landmark(numStart, numEnd, imageFolder, outputFolder)

# Step 2: Morph faces
    morphsFolder = "./images/Morphs/"   # stores morphs with black background
    # faceMorph(numStart, numEnd, imageFolder, outputFolder, morphsFolder)

# Step 3: Get facial landmarks of morph
    morphOutputFolder = "./landmarks/Impostors_M_Output/"
    # landmark_morph(numStart, numEnd, morphsFolder, morphOutputFolder)
 
# Step 4: Change background of morphs to skin
    kMorphsFolder = "./images/Morphs_K/"
    k_skin(numStart, numEnd, morphsFolder, kMorphsFolder) 

# Step 5: Face swap k_morph onto original pairs
    finalFolder = "./images/Swapped_Morphs_K/"
    flag = "k"
    # faceSwap(numStart, numEnd, kMorphsFolder, imageFolder, morphOutputFolder, outputFolder, finalFolder, flag) 

# Alternate Step 4 1/2: Remove sideburns
    newOutputFolder = "./landmarks/Adjusted_Output/"
    remove_sideburns(numStart, numEnd, outputFolder, newOutputFolder)

# TODO:
# Alterate Step 5: Face swap + Laplacian + skin method
    finalFolder = "./images/Swapped_Morphs_Lap/" 
    laplacian_swap(numStart, numEnd, kMorphsFolder, imageFolder, morphOutputFolder, newOutputFolder, finalFolder)

# Original morph method (no skin background)
#    finalFolder = "./images/Swapped_Morphs/"
#    flag = "none"
#    faceSwap(numStart, numEnd, morphsFolder, imageFolder, morphOutputFolder, outputFolder, finalFolder)

# Laplacian method (attempt)
#    finalFolder = "./old_versions/Laplacian/"
#    faceSwap_lap(numStart, numEnd, morphsFolder, imageFolder, morphOutputFolder, outputFolder, finalFolder)
