# KITTI TRACKING BENCHMARK DEMONSTRATION

# This tool displays the images and the object labels for the benchmark. 
# Before running this tool, set root_dir to the directory where you have
# downloaded the dataset. 'root_dir' must contain the subdirectory
# 'training', which in turn contains 'image_02', 'label_02' and 'calib'.


import numpy as np
from numpy import array 
import cv2 
import os


#=============================================================================================
	#...This finds the number of lablled object in an image
	#...here we are considering 1st image

	#...............FORMAT OF TRACKLETS.......................#
	#.....tracklets[A][B][C]
	#.....A is the image number
	#.....B is infact for looping through all the 40 (assumed upper limit for labels)
	#.....C is for accesing the field :'type'   

def find_count(passed_tracklets, idx):
	count = 0
	for i in range(150):
		#3rd argument is t3e labelled object name
		if passed_tracklets[idx][i][2] != 0:
			count = count + 1

	return count	 		

#=========================================================================================

def readCalibration(calib_dir, seq_idx, cam, data_type):

	if data_type == 'kitti':
		fid = open(os.path.join(calib_dir, '%04d.txt' % seq_idx), "r")
	elif data_type == 'nuscenes':
		fid = open(os.path.join(calib_dir, 'scene-' + '%04d.txt' % seq_idx), "r")
	else:
		fid = None

	
	#Uncomment the below two lines to print the file contents on terminal

	#...........................................................................#
	
	#    We need to access only the first 4 lines in calib files. So create a file temp.txt 
	#    to store the first 4 lines. readlines() read the file line by line and store each 
	#    line as an element of a list.As we need just the first 4 line we run the loop four 
	#    time and write each line into the temporary text file.Now close the temporary file 
	#    and open it again but in the read mode .

	temp_file = open('temp.txt', 'w')
	lines = fid.readlines()
	for var in range(4):
		temp_file.write(lines[var] + '\n')  	#go to next line after writing a line

	temp_file.close()
	temp = open('temp.txt', 'r')

	C = np.loadtxt(temp, delimiter=' ', dtype={'names': ('col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7',
														 'col8', 'col9', 'col10', 'col11', 'col12', 'col13'),
											   'formats': ('S3', 'float', 'float', 'float', 'float', 'float', 'float',
														   'float', 'float', 'float', 'float', 'float', 'float')})


	# Since we are using only the second camera we only need the P2 matrix
	# typecasting first index of P to int is necesssary, else that is float and error happens..BEWARE..!!

	P = np.zeros((3, 4))
	for i in range(12):
		P[int(np.floor(i / 4))][np.mod(i, 4)] = C[cam][i+1]

	# delete the temporary file

	os.remove('temp.txt')
	fid.close()
	return P

#===========================================================================================

def readLabels(label_dir, seq_idx, nimages, lable_root=None):

	if lable_root is None:
		labelfile = os.path.join(label_dir, '%04d.txt' % seq_idx)
	else:
		labelfile = lable_root
	fid = open(labelfile, "r")

	lines = fid.readlines()
	line_0_stripped = lines[0].strip()

	# No idea why +2 is working. When we tried +1 the last coloumn of the file was'nt getting read..!!! 
	
	num_cols = line_0_stripped.count(' ') + 2
	#print 'Num of cols' + str(num_cols)

	fid.close()

	fid = open(labelfile, "r")

	try:
		if num_cols == 17:
			C = np.loadtxt(fid, delimiter=' ',
					dtype={'names': ('col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9',
									 'col10', 'col11', 'col12', 'col13''col14', 'col15', 'col16', 'col17'),
						   'formats': ('float', 'float', 'S10', 'float', 'float', 'float', 'float', 'float', 'float',
									   'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float')})

		elif num_cols == 18:
			C = np.loadtxt(fid, delimiter=' ',
					dtype={'names': ('col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9',
									 'col10', 'col11', 'col12', 'col13''col14', 'col15', 'col16', 'col17', 'col18'),
				    	   'formats': ('float', 'float', 'S10', 'float', 'float', 'float', 'float', 'float', 'float',
									   'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float'
									   )})

		elif num_cols == 19:
			C = np.loadtxt(fid, delimiter=' ',
					dtype={'names': ('col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10',
									 'col11', 'col12', 'col13''col14', 'col15', 'col16', 'col17', 'col18', 'col19'),
				    	   'formats': ('float', 'float', 'S10', 'float', 'float', 'float', 'float', 'float', 'float',
									   'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
									   'float')})


		else:
			print('Else : This file is not in KITTI tracking format.')

	except:
		print('Except : This file is not in KITTI tracking format.')

	fid.close()

	# for future references for finding nimages without passing as param
	#print int(max(zip(*C)[0]))+1

	unzipped_version = list(zip(*C))
	#print unzipped_version

	#array_version = array(unzipped_version)
	#print array_version.shape
	#shape of array_version is (16,4271)
	
	np_array = np.array(unzipped_version[0])


	# The number 40 is selected considering that number of labelled objects in an image would
	# above 40. TRADING SPACE FOR TIME...NOT REALLY TECHNICAL..!!!

	final_data = [[[0 for x in range(num_cols)] for y in range(150)] for z in range(nimages)]
	# print array(final_data).shape

	# In the thrid loop (num_cols - 1) is used because num_cols is 18 , but as much as we are concerned
	# there are only 17 columns present and when we print also only 17 are getting printed...STRANGE.!! 

	for img in range(nimages):
		index, = np.where(np_array == img)    # COMMA IS PUT DELIBERATELY TO DIRECTLY GET INDICES
		for i in range(len(index)):
			annotated_obj = index[i]
			# print annotated_obj
			#Infact annotated_obj varies from 0 to 4270
			for j in range(num_cols - 1):
				final_data[img][i][j] = unzipped_version[j][annotated_obj]
			
	return final_data

#============================================================================================


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def drawBox2D(image, passed_object):

	color_i = get_color(int(passed_object[1]))
	cv2.rectangle(image, (int(passed_object[6]), int(passed_object[7])), (int(passed_object[8]), int(passed_object[9])),
				  color_i, 2)
	tl = 2
	cv2.putText(image, '{}{}{}'.format(passed_object[2], ':', int(passed_object[1])),
				(int(passed_object[6]), int(passed_object[7]) - 5), 0, tl / 3, color_i,
				thickness=2, lineType=cv2.LINE_AA)
	# if passed_object[2] == b'DontCare':
	# 	cv2.rectangle(image, (int(passed_object[6]), int(passed_object[7])), (int(passed_object[8]), int(passed_object[9])), (103, 192, 255), 1)
	# elif passed_object[2] == b'Car':
	# 	cv2.rectangle(image, (int(passed_object[6]), int(passed_object[7])), (int(passed_object[8]), int(passed_object[9])), (0, 0, 255), 2)
	# elif passed_object[2] == b'Van':
	# 	cv2.rectangle(image, (int(passed_object[6]), int(passed_object[7])), (int(passed_object[8]), int(passed_object[9])), (255, 0, 0), 1)
	# elif passed_object[2] == b'Truck':
	# 	cv2.rectangle(image, (int(passed_object[6]), int(passed_object[7])), (int(passed_object[8]), int(passed_object[9])), (255, 255, 0), 1)
	# elif passed_object[2] == b'Pedestrian':
	# 	cv2.rectangle(image, (int(passed_object[6]), int(passed_object[7])), (int(passed_object[8]), int(passed_object[9])), (0, 255, 0), 1)
	# elif passed_object[2] == b'Person_sitting':
	# 	cv2.rectangle(image, (int(passed_object[6]), int(passed_object[7])), (int(passed_object[8]), int(passed_object[9])), (0, 255, 255), 1)
	# elif passed_object[2] == b'Cyclist':
	# 	cv2.rectangle(image, (int(passed_object[6]), int(passed_object[7])), (int(passed_object[8]), int(passed_object[9])), (255, 0, 255), 1)
	# elif passed_object[2] == b'Tram':
	# 	cv2.rectangle(image, (int(passed_object[6]), int(passed_object[7])), (int(passed_object[8]), int(passed_object[9])), (155, 155, 155), 1)
	# elif passed_object[2] == b'Misc':
	# 	cv2.rectangle(image, (int(passed_object[6]), int(passed_object[7])), (int(passed_object[8]), int(passed_object[9])), (128, 105, 255), 1)

def drawBox3D(image, qs, passed_object, thickness=2):

	""" Draw 3d bounding box in image
		qs: (8,3) array of vertices for the 3d box in following order:
			1 -------- 0
		   /|         /|
		  2 -------- 3 .
		  | |        | |
		  . 5 -------- 4
		  |/         |/
		  6 -------- 7
	"""
	# if passed_object[2] == b'DontCare':
	# 	color = (103, 192, 255)
	# elif passed_object[2] == b'Car':
	# 	color = (0, 0, 255)
	# elif passed_object[2] == b'Van':
	# 	color = (255, 0, 0)
	# elif passed_object[2] == b'Truck':
	# 	color = (255, 255, 0)
	# elif passed_object[2] == b'Pedestrian':
	# 	color = (0, 255, 0)
	# elif passed_object[2] == b'Person_sitting':
	# 	color = (0, 255, 255)
	# elif passed_object[2] == b'Cyclist':
	# 	color = (255, 0, 255)
	# elif passed_object[2] == b'Tram':
	# 	color = (155, 155, 155)
	# elif passed_object[2] == b'Misc':
	# 	color = (128, 105, 255)

	color = get_color(int(passed_object[1]))
	qs = qs.astype(np.int32)

	tl = 2
	support_classes = [b'Pedestrian', b'Car', b'pedestrian', b'car', b'bicycle', b'truck', b'motorcycle', b'bus',
					   b'trailer']
	if passed_object[2] in support_classes:
		cv2.putText(image, '{}'.format(int(passed_object[1])), (int(qs[0, 0]), int(qs[0, 1]) - 10), 0, tl / 3, color,
					thickness=2, lineType=cv2.LINE_AA)

		for k in range(0, 4):
			# Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
			i, j = k, (k + 1) % 4
			cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
			i, j = k + 4, (k + 1) % 4 + 4
			cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
			i, j = k, k + 4
			cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

	return image


#============================================================================================


def projectToImage(passed_corners_3D, passed_P):

	np_passed = array(passed_corners_3D)
	num_cols_passed_corners_3D = np_passed[1].size

	pts = np.append(np_passed, [np.ones(num_cols_passed_corners_3D)], axis=0)
	pts_2D = np.matmul(passed_P, pts)

	for i in range(3):
		pts_2D[0][i] = pts_2D[0][i] / pts_2D[2][i]
		pts_2D[1][i] = pts_2D[1][i] / pts_2D[2][i]

	np_pts_2D = array(pts_2D)
	ret_pts_2D = np.delete(np_pts_2D, (2), axis=0)

	return ret_pts_2D


def project_to_image(pts_3d, P):
	""" Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    """
	n = pts_3d.shape[0]
	pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
	pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
	pts_2d[:, 0] /= pts_2d[:, 2]
	pts_2d[:, 1] /= pts_2d[:, 2]
	return pts_2d[:, 0:2]


#============================================================================================	


def computeBox3D(passed_tracklets, P):
	""" Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
	R = [[+np.cos(passed_tracklets[16]), 0, + np.sin(passed_tracklets[16])],
		 [0, 1, 0],
		 [-np.sin(passed_tracklets[16]), 0, +np.cos(passed_tracklets[16])]]

	# 3d bounding box dimensions
	l = passed_tracklets[12]
	w = passed_tracklets[11]
	h = passed_tracklets[10]

	# 3d bounding box corners
	x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
	y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
	z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

	# rotate and translate 3d bounding box
	corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
	corners_3d[0, :] = corners_3d[0, :] + passed_tracklets[13]
	corners_3d[1, :] = corners_3d[1, :] + passed_tracklets[14]
	corners_3d[2, :] = corners_3d[2, :] + passed_tracklets[15]

	# only draw 3d bounding box for objs in front of the camera
	if np.any(corners_3d[2, :] < 0.1):
		corners_2d = None
		return corners_2d, np.transpose(corners_3d)

	# project the 3d bounding box into the image plane
	corners_2d = project_to_image(np.transpose(corners_3d), P)

	return corners_2d, np.transpose(corners_3d)


#============================================================================================


if __name__ == "__main__":

	print("======= KITTI Development Kit Demo =======")

	root_dir = '~/nusc_kitti'
	# data_set = 'training'
	data_set = 'mini_train'
	train__dir = 'image_02'

	data_type = 'nuscenes'

	# set camera
	cam = 2	 # 2 = left color camera

	full_path = os.path.join(root_dir, data_set, train__dir)

	sub_dir = os.listdir(full_path)

	num_sequences = len(sub_dir) - 2
	
	# for kitti
	# seq_idx = 16
	# image_dir = os.path.join(root_dir, data_set, 'image_02/0016')
	
	# for nuscenes
	seq_idx = 61
	image_dir = os.path.join(root_dir, data_set, 'image_02/scene-0061')
	label_root = os.path.join(root_dir, data_set, 'label_02/scene-' + '%04d.txt' % seq_idx)

	label_dir = os.path.join(root_dir, data_set, 'label_02')
	calib_dir = os.path.join(root_dir, data_set, 'calib')

	P = readCalibration(calib_dir, seq_idx, cam, data_type)
	print('P is' + str(P))

	nimages = len(os.listdir(image_dir))

	tracklets = readLabels(label_dir, seq_idx, nimages, label_root)

	img0 = tracklets[1]

	for img_idx in range(nimages):
		count = find_count(tracklets, img_idx)

		image = cv2.imread(os.path.join(image_dir, '%06d.png' % img_idx))
		img1 = np.copy(image)  # for 2d bbox
		img2 = np.copy(image)  # for 3d bbox
		# cv2.imshow
		for obj_idx in range(count):
			#for drawing 2D bounding boxes
			drawBox2D(img1, tracklets[img_idx][obj_idx])
			#for drawing 3D bounding boxes
			box3d_pts_2d, _ = computeBox3D(tracklets[img_idx][obj_idx], P)
			if box3d_pts_2d is None:
				print("something wrong with image_{} of the 3D box.".format(img_idx))
				continue
			else:
				drawBox3D(img2, box3d_pts_2d, tracklets[img_idx][obj_idx], thickness=2)
		# cv2.imshow('3d_show', img2)
		cv2.imwrite("2d_image_" + "%04d.jpg" % img_idx, img1)
		cv2.imwrite("3d_image_" + "%04d.jpg" % img_idx, img2)
		cv2.waitKey(50)


	if(cv2.waitKey(0) & 0xFF == ord('q')):
		cv2.destroyAllWindows()
