#Face Morph Research#

The objective of this research project is to investigate the "Magic Passport" Security breach
as outlined in Matteo Ferrara, Annalisa Franco, and David Maltoni's publication from 2014.

https://www.researchgate.net/publication/283473746_The_magic_passport


##Instructions##
The pairs of faces to be morphed are found in the **pairs** folder. These images are labelled 
numerically, #a, #b to facilitate automatic facial landmark detection.

landmark.py uses dlib and the shape predictor 68 dataset to generate .txt files of 68 facial landmarks
for each image. These .txt files are saved to the **output** folder.
Run landmark.py with the command 'python landmark.py'

faceMorph.py creates the face morphs and outputs them to the **morphs** folder. Run faceMorph.py with the command 'python faceMorph.py'

faceSwap.py pastes the image morph mask onto one of the original image pairs. The default image
choice is #a. Run faceSwap.py with the command 'python faceSwap.py'
Currently faceSwap is set to only swap the 3rd pair of images. 
 
landmark2.py, faceMorph2.py, and faceSwap2.py are alternate versions of 


Credits:
Credits for faceMorph.py and faceSwap.py go to Satya Mallick's face-swap and face-morph tutorials, found here:
https://www.learnopencv.com/face-swap-using-opencv-c-python/
https://www.learnopencv.com/face-morph-using-opencv-cpp-python/

