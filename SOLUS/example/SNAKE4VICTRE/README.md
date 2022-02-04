README file

Generation of dcm file and segmentation procedure for k-wave US-bmode images obtained from victre-based acoustic digital phantom.

To generate a prior based on the DT transform, the genSnakePrior.m file should be amended with the input b-mode image and name of the output file. 

The script requires the installation of the software suite SOLUS at https://github.com/andreafarina/SOLUS

After the script is launched a DICOM file is generated. This is loaded to start the semi-autmatic segmentation. The user will click few points on the borders of the lesion to segment. 
After this the snake fitting will find the best approximation for the segmentation and will save a mat file suitable for optical reconstruction. 
