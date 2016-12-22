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

def createboundingbox(records):
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
									[imagedit, imagename] = findimageloc(timestamp, recnum,playlistname, now["video_id"])
									if (now["object_label"]["label"][corr]["objectSelect"] == None):
										continue
									if now["object_ids"][corr] not in objectids:
										print now["object_ids"][corr]
										objectids[now["object_ids"][corr]] = 1
									else:
										objectids[now["object_ids"][corr]] += 1
									if objectids[now["object_ids"][corr]] > 20:
										continue
									makeboximage(imagedit, str(now["object_ids"][corr]) + "_"+ playlistname + "_" + str(now["video_id"]) + "_" + imagename, box[0]/2, box[1]/2, box[2]/2, box[3]/2)
									
def findimageloc(timestamp, recnum, playlistname, video_id):
	files = [p for p in os.listdir('/home/amr6114/alexannotation/'+ playlistname+ '/d_parsed/00'+ str(video_id)+'/')]
	files = sorted(files)
	filewanted = '/home/amr6114/alexannotation/'+ playlistname+ '/d_parsed/00'+ str(video_id)+'/'+ files[timestamp]
	return filewanted, files[timestamp]

def makeboximage(image, imagename, x0, y0, x1, y1):
    # get an image
    base = Image.open(image)
 
    # get a drawing context
    d = ImageDraw.Draw(base)
    d.rectangle([x0, y0, x1, y1], None, 255)
    del d

    print "here33"

    base.save('/home/amr6114/uploadimages/'+imagename)

def main(params):
    # Parse xml dump into dict
    records = get_records(params["root_dir"])

    # Convert records to new version
    createboundingbox(records)


if __name__ == "__main__":                                                                                    
    parser = argparse.ArgumentParser()
    parser.add_argument('-root_dir', default='/home/amr6114/alexannotation', type=str)
    parser.add_argument('-video_length', default=199, type=int)

    args = vars(parser.parse_args())
    main(args)
