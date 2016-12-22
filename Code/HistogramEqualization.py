import cv2
import os
import sys
import glob


filenames = glob.glob('/afs/cs.stanford.edu/u/amr6114/cropped/*.jpg')
for filen in filenames:
	img = cv2.imread(filen,0)
	equ = cv2.equalizeHist(img)
	cv2.imwrite('/afs/cs.stanford.edu/u/amr6114/contrastim1/' + os.path.basename(filen),equ)
