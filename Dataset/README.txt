The images are proprietary from the local hospital and I donÕt have the ability share them. The dataset had two folders: Yes and No, where each frame was put into the respective folders. The annotation system from the previous researcher had an output text file with the following columns:

Columns (all values are serialized as int64)
0: Bounding box, top left X
1: Bounding box, top left y
2: Bounding box, bot right x
3: Bounding box, bot right y
4: Detection ID, refers to the unique track ID (-1 if no person, 1 if person)
5: Action class
6: Frame ID (frame ID = 1039 in the filename e.g. d-1039.jpg)
7: Timestamp (comes from timestamps-8-28.txt)
8: Camera ID (last part of the IP address)

The process of converting the text file to be applicable to this project is in the folder Annotation System. An example of the text file is given as previous.txt. The json created for this project had a python readable set notation. An example of notation is given as records.json. 
