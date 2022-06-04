import mediapipe as mp
import cv2
import numpy as np


if __name__ == "__main__":

	#姿勢推定
	mp_pose = mp.solutions.pose

	#動画読み込み
	cap_file = cv2.VideoCapture('training.mp4')

	with mp_pose.Pose(
        min_detection_confidence=0.5,
        static_image_mode=False) as pose_detection:

		while cap_file.isOpened:
			success, image = cap_file.read()

			if not success:
				print("empty camera frame")
				break

			#1枚の画像の大きさをリサイズする
			image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5)
			rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			height = rgb_image.shape[0]
			width = rgb_image.shape[1]

			results = pose_detection.process(rgb_image)
			cv2.imshow('AI personal trainer', image)

			if cv2.waitKey(5) & 0xFF == 27:
				break