from absl import app, flags, logging
from absl.flags import FLAGS
import time
import cv2
import os
import glob
import pathlib
import numpy as np
import tensorflow as tf

from api import PRN
from modules.utils import load_yaml, set_memory_growth
from modules.cv_plot import plot_kpt, plot_vertices, plot_pose_box
from modules.estimate_pose import estimate_pose

flags.DEFINE_boolean('use_cam', False, 'demo with webcam')
flags.DEFINE_string('cfg_path', './configs/prnet.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', './Data/test-img', 'path to input image')
flags.DEFINE_string('save_path', './Data/save-img', 'path to save result')


def main(_):
	# init
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
	set_memory_growth()

    # load PRNet model
	cfg = load_yaml(FLAGS.cfg_path)
	model = PRN(cfg, is_dlib=True)

	# evaluation
	if not FLAGS.use_cam:  # on test-img
		print("[*] Processing on images in {}. Press 's' to save result.".format(FLAGS.img_path))
		img_paths = glob.glob(os.path.join(FLAGS.img_path, '*'))
		for img_path in img_paths:
			img = cv2.imread(img_path)
			pos = model.process(img_path)
			if pos is None:
				continue

			vertices = model.get_vertices(pos)
			kpt = model.get_landmarks(pos)
			camera_matrix, _ = estimate_pose(vertices)

			cv2.imshow('Input', img)
			cv2.imshow('Sparse alignment', plot_kpt(img, kpt))
			cv2.imshow('Dense alignment', plot_vertices(img, vertices))
			cv2.imshow('Pose', plot_pose_box(img, camera_matrix, kpt))
			cv2.moveWindow('Input', 0, 0)
			cv2.moveWindow('Sparse alignment', 500, 0)
			cv2.moveWindow('Dense alignment', 1000, 0)
			cv2.moveWindow('Pose', 1500, 0)

			key = cv2.waitKey(0)
			if key == ord('q'):
				exit()
			elif key == ord('s'):
				cv2.imwrite(os.path.join(FLAGS.save_path, 
					os.path.basename(img_path)), plot_kpt(img, kpt))
				print("Result saved in {}".format(FLAGS.save_path))

	
	else:  # webcam demo
		cap = cv2.VideoCapture(0)
		start_time = time.time()
		count = 1
		while(True):
			_, image = cap.read()
			
			pos = model.process(image)
			fps_str = 'FPS: %.2f' % (1 / (time.time() - start_time))
			start_time = time.time()
			cv2.putText(image, fps_str, (25, 25),
						cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)
			cv2.imshow('Input', image)
			cv2.moveWindow('Input', 0, 0)

			key = cv2.waitKey(1)
			if pos is None:
				cv2.waitKey(1)
				cv2.destroyWindow('Sparse alignment')
				cv2.destroyWindow('Dense alignment')
				cv2.destroyWindow('Pose')
				if key & 0xFF == ord('q'):
					break
				continue

			else:
				vertices = model.get_vertices(pos)
				kpt = model.get_landmarks(pos)
				camera_matrix, _ = estimate_pose(vertices)

				result_list = [plot_kpt(image, kpt),
							   plot_vertices(image, vertices),
							   plot_pose_box(image, camera_matrix, kpt)]
				
				cv2.imshow('Sparse alignment', result_list[0])
				cv2.imshow('Dense alignment', result_list[1])
				cv2.imshow('Pose', result_list[2])
				cv2.moveWindow('Sparse alignment', 500, 0)
				cv2.moveWindow('Dense alignment', 1000, 0)
				cv2.moveWindow('Pose', 1500, 0)

				if key & 0xFF == ord('s'):
					image_name = 'prnet_cam_' + str(count)
					save_path = FLAGS.save_path

					cv2.imwrite(os.path.join(
						save_path, image_name + '_result.jpg'), 
						np.concatenate(result_list, axis=1))
					cv2.imwrite(os.path.join(
						save_path, image_name + '_image.jpg'), image)
					count += 1
					print("Result saved in {}".format(FLAGS.save_path))

				if key & 0xFF == ord('q'):
					break

if __name__ == '__main__':
	try:
		app.run(main)
	except SystemExit:
		pass
