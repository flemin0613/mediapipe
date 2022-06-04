import mediapipe as mp
import cv2
import numpy as np


#画像上の各骨格情報を取得する関数
def get_keypoint(results, height, width):
	#キーポイントの座標の大きさは、0～１に正規化されているため、大きさを戻す

	#左肩
	left_shoulder_x = int(results.pose_landmarks.landmark[11].x * width)
	left_shoulder_y = int(results.pose_landmarks.landmark[11].y * height)
	left_shoulder_xy = [left_shoulder_x, left_shoulder_y]

	#左肘
	left_elbow_x = int(results.pose_landmarks.landmark[13].x * width)
	left_elbow_y = int(results.pose_landmarks.landmark[13].y * height)
	left_elbow_xy = [left_elbow_x, left_elbow_y]

	#左手首
	left_wrist_x = int(results.pose_landmarks.landmark[15].x * width)
	left_wrist_y = int(results.pose_landmarks.landmark[15].y * height)
	left_wrist_xy = [left_wrist_x, left_wrist_y]

	#左おしり
	left_hip_x = int(results.pose_landmarks.landmark[23].x * width)
	left_hip_y = int(results.pose_landmarks.landmark[23].y * height)
	left_hip_xy = [left_hip_x, left_hip_y]

	#左膝
	left_knee_x = int(results.pose_landmarks.landmark[25].x * width)
	left_knee_y = int(results.pose_landmarks.landmark[25].y * height)
	left_knee_xy = [left_knee_x, left_knee_y]

	#左足首
	left_ankle_x = int(results.pose_landmarks.landmark[27].x * width)
	left_ankle_y = int(results.pose_landmarks.landmark[27].y * height)
	left_ankle_xy = [left_ankle_x, left_ankle_y]

	return left_shoulder_xy, left_elbow_xy, left_wrist_xy, left_hip_xy, left_knee_xy, left_ankle_xy


def calc_distance(x1, y1, x2, y2, x3, y3):

	#uベクトル
	#(x1,y1) → (x2,y2)

	#vベクトル
	#(x1,y1) → (x3,y3)

    u = np.array([x2 - x1, y2 - y1])
    v = np.array([x3 - x1, y3 - y1])

	# np.cross(u,v) → uベクトルとvベクトルの外積
	# np.linalg.norm(u) → uベクトルの長さ
    L = abs(np.cross(u, v)/np.linalg.norm(u))
    return L

#肩と足首の2点の傾きを取得
def calc_slope(left_shoulder_xy, left_ankle_xy):
    x = [left_shoulder_xy[0], left_ankle_xy[0]]
    y = [left_shoulder_xy[1], left_ankle_xy[1]]

	#2点の傾きと切片を取得 
	#np.polyfit(x,y,1) 1は1次式
    slope, intercept = np.polyfit(x,y,1)
    return slope

#現在の骨格情報から、腕立てができていることをフラグで返す
def get_low_pose(THRESH_SLOPE, THRESH_DIST_SPINE, THRESH_ARM, flg_low, slope, dist_hip, dist_knee, dist_elbow):
    if slope <= THRESH_SLOPE and dist_hip < THRESH_DIST_SPINE and \
                    dist_knee < THRESH_DIST_SPINE and dist_elbow > THRESH_ARM:
        flg_low = True
    else:
        flg_low = False
    return flg_low


#骨格情報を描画する処理
def draw_keypoint(image, RADIUS, CLR_KP, CLR_LINE, THICKNESS, left_shoulder_xy, \
                  left_elbow_xy, left_wrist_xy, left_hip_xy, left_knee_xy, left_ankle_xy):

	#肩
    cv2.circle(image, (left_shoulder_xy[0], left_shoulder_xy[1]), RADIUS, CLR_KP, THICKNESS)
	#肘
    cv2.circle(image, (left_elbow_xy[0], left_elbow_xy[1]), RADIUS, CLR_KP, THICKNESS)
	#手首
    cv2.circle(image, (left_wrist_xy[0], left_wrist_xy[1]), RADIUS, CLR_KP, THICKNESS)
	#お尻
    cv2.circle(image, (left_hip_xy[0], left_hip_xy[1]), RADIUS, CLR_KP, THICKNESS)
	#膝
    cv2.circle(image, (left_knee_xy[0], left_knee_xy[1]), RADIUS, CLR_KP, THICKNESS)
	#足首
    cv2.circle(image, (left_ankle_xy[0], left_ankle_xy[1]), RADIUS, CLR_KP, THICKNESS)

	#肩と肘を線で描画
    cv2.line(image, (left_shoulder_xy[0], left_shoulder_xy[1]), (left_elbow_xy[0], left_elbow_xy[1]), CLR_LINE, THICKNESS, lineType=cv2.LINE_8, shift=0)
	#肘と手首を線で描画
    cv2.line(image, (left_elbow_xy[0], left_elbow_xy[1]), (left_wrist_xy[0], left_wrist_xy[1]), CLR_LINE, THICKNESS, lineType=cv2.LINE_8, shift=0)
	#肩とお尻を線で描画
    cv2.line(image, (left_shoulder_xy[0], left_shoulder_xy[1]), (left_hip_xy[0], left_hip_xy[1]), CLR_LINE, THICKNESS, lineType=cv2.LINE_8, shift=0)
	#お尻と膝を線で描画
    cv2.line(image, (left_hip_xy[0], left_hip_xy[1]), (left_knee_xy[0], left_knee_xy[1]), CLR_LINE, THICKNESS, lineType=cv2.LINE_8, shift=0)
	#膝と足首を線で描画
    cv2.line(image, (left_knee_xy[0], left_knee_xy[1]), (left_ankle_xy[0], left_ankle_xy[1]), CLR_LINE, THICKNESS, lineType=cv2.LINE_8, shift=0)

    return image



if __name__ == "__main__":


	#肩と足首の傾きが0に近ければ、低姿勢
	THRESH_SLOPE = 0

	#肩と足首の直線に対して、お尻と膝の距離が30以内であれば、低姿勢
	THRESH_DIST_SPINE = 30

	#肩と手首の直線に対して、肘の距離が40以上であれば、腕を十分に曲げることができている
	THRESH_ARM = 40

	#縁の半径を5
	RADIUS = 5
	#線の太さ2
	THICKNESS = 2
	#座標の色
	CLR_KP = (0, 0, 255)
	#線の色白
	CLR_LINE = (255, 255, 255)


	#姿勢推定
	mp_pose = mp.solutions.pose

	#動画読み込み
	cap_file = cv2.VideoCapture('training.mp4')

	with mp_pose.Pose(
        min_detection_confidence=0.5,
        static_image_mode=False) as pose_detection:

		count = 0
		flg_low = False

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

			if not results.pose_landmarks:
				print('not results')
			else:

				#推定に応じた骨格情報を取得
				#1 :肩
				#2 :肘
				#3 :手首
				#4 :おしり　垂直の直線を求める
				#5 :ひざ　垂直の直線を求める
				#6 :足首
				# 動画上は、左半身のみ映っているため、左半身の座標のみ利用する

				#各骨格情報取得
				left_shoulder_xy, left_elbow_xy, left_wrist_xy, left_hip_xy, left_knee_xy, left_ankle_xy = get_keypoint(results, height, width)

				#肩と足首の直線に対して、お尻の座標の垂直な直線距離を求める
				dist_hip = calc_distance(left_shoulder_xy[0], left_shoulder_xy[1], left_ankle_xy[0], left_ankle_xy[1], left_hip_xy[0], left_hip_xy[1])

				#肩と足首の直線に対して、膝の座標の垂直な直線距離を求める
				dist_knee = calc_distance(left_shoulder_xy[0], left_shoulder_xy[1], left_ankle_xy[0], left_ankle_xy[1], left_knee_xy[0], left_knee_xy[1])

				#肩と手首の直線に対して、肘の座標の垂直な直線距離を求める
				dist_elbow = calc_distance(left_shoulder_xy[0], left_shoulder_xy[1], left_wrist_xy[0], left_wrist_xy[1], left_elbow_xy[0], left_elbow_xy[1])

				#肩と足首の傾きを取得
				body_slope = calc_slope(left_shoulder_xy, left_ankle_xy)

				#腕立て状態を保持しておく
				pre_flg_low = flg_low

				#現在の時点の腕立て状態を取得
				flg_low = get_low_pose(THRESH_SLOPE, THRESH_DIST_SPINE, THRESH_ARM, \
						flg_low, body_slope, dist_hip, dist_knee, dist_elbow)

				#1フレーム前の腕立て状態がFALSE かつ 現在の腕立て状態がTRUE
				if pre_flg_low == False and flg_low == True:
					count += 1

				#腕立て回数を画面描画
				#位置は20,100
				cv2.putText(image, str(int(count)), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
				image = draw_keypoint(image, RADIUS, CLR_KP, CLR_LINE, THICKNESS, \
					left_shoulder_xy, left_elbow_xy, left_wrist_xy, \
					left_hip_xy, left_knee_xy, left_ankle_xy)


			cv2.imshow('AI personal trainer', image)

			if cv2.waitKey(5) & 0xFF == 27:
				break


	print('合計'+str(count)+'回')

cap_file.release()
cv2.destroyAllWindows()
