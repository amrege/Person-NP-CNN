import csv
import os
import argparse
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from parse_labels import read_json, write_json
from PIL import Image, ImageDraw
from scipy import misc

def get_records(data_dir):
	playlists = [p for p in os.listdir(data_dir) if p.find("10.") is not -1]
	records = {}
	for p in playlists:
		infile = os.path.join(data_dir, p, "records.json")
		if os.path.exists(infile):
			record_dict = read_json(infile)
			records[p] = record_dict["records"]
	return records

def createboundingbox(records, validpple):
	for playlistname in records:
		if playlistname not in ["10.0.1.4"]:
			continue
		objectids = {}
		# Creates a dict w/ frame label (empty), video id (int), playlist name (string), object label (each obj has bbox equiv), bboxes
		recnum = 0
		for now in records[playlistname]:
			recnum += 1
			if (len(now["bboxes"]["label"]) != len(now["object_label"]["label"])):
				continue
			for corr in xrange(0, len(now["bboxes"]["label"])):
				bboxes = now["bboxes"]["label"][corr]
				objlabel = now["object_label"]["label"][corr]
				# each box listing should be relatable to object listing 
				for (boxk, boxo), (objk, objo) in zip(bboxes.items(),objlabel.items()):
					for onoroff in objo:
						if onoroff[2] == 0:
							for box in boxo:
								# Check if box is valid and the peron is on screen for the event
								if(onoroff[0] <= box[4] and box[4] <= onoroff[1]):
									timestamp = box[4]
									[imagedit, imagename] = findimageloc(timestamp, recnum,playlistname)
									if (now["object_label"]["label"][corr]["objectSelect"] == None):
										continue
									print now["object_ids"][corr]
									if str(now["object_ids"][corr]) not in validpple[playlistname]:
										print validpple[playlistname]
										continue
									makeboximage(imagedit, str(now["object_ids"][corr]) + "_"+ playlistname + "_" + imagename, box[0]/2, box[1]/2, box[2]/2, box[3]/2)
									
def findimageloc(timestamp, recnum, playlistname):
	filedirectories = [p for p in os.listdir('/home/amr6114/alexannotation/'+ playlistname+ '/d_parsed/')]
	filedirectories = sorted(filedirectories)
	filed = filedirectories[recnum-1]
	begin = 0
	end = len([p for p in os.listdir('/home/amr6114/alexannotation/'+ playlistname+ '/d_parsed/'+ filed+'/')])
	if timestamp >= begin and timestamp < end:
		files = [p for p in os.listdir('/home/amr6114/alexannotation/'+ playlistname+ '/d_parsed/'+ filed+'/')]
		files = sorted(files)
		filewanted = '/home/amr6114/alexannotation/'+ playlistname+ '/d_parsed/'+ filed+'/'+ files[timestamp-begin]
		return filewanted, files[timestamp-begin]
		countindex = end

def makeboximage(image, imagename, x0, y0, x1, y1):
    # get an image
    base = Image.open(image)
 
    # get a drawing context
    d = ImageDraw.Draw(base)
    d.rectangle([x0, y0, x1, y1], None, 255)
    del d


    base.save('/home/amr6114/secondround/'+imagename)

def readcsvandparse(filename):
    playlistann = {}
    with open(filename, 'rb') as csvfile:
	annotations = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in annotations:
		rowbreak = row[0].split(',')
		nameofimage = rowbreak[0]
		breakdown = nameofimage.split('_')
		classofbb = int(rowbreak[4]) - int(rowbreak[3])
		if classofbb > 0:
			classofbb = 1
		if classofbb < 0:
			classofbb = -1
		if breakdown[1] not in playlistann: #for each playlist have a dict of the person classifications
			personids = {}
			personids[breakdown[0]] = classofbb
			playlistann[breakdown[1]] = personids
		else:# if playlist exists
			if breakdown[0] in personids: # check if id already exists
				playlistann[breakdown[1]][breakdown[0]] += classofbb
			else:
				playlistann[breakdown[1]][breakdown[0]] = classofbb
    
    for playlist in playlistann:
	objectremove = []
	for objid in playlistann[playlist]:
		if playlistann[playlist][objid] <= 0:
			print objid
			objectremove.append(objid)
	for obj in objectremove:
		del playlistann[playlist][obj]
    return playlistann

	
def main(params):
    pple = readcsvandparse('/home/amr6114/alexannotation/personnotperson.csv')
    print pple
    # Parse xml dump into dict
    records = get_records(params["root_dir"])

    # Convert records to new version
    createboundingbox(records, pple)


if __name__ == "__main__":                                                                                    
    parser = argparse.ArgumentParser()
    parser.add_argument('-root_dir', default='/home/amr6114/alexannotation', type=str)
    parser.add_argument('-video_length', default=199, type=int)

    args = vars(parser.parse_args())
    main(args)
