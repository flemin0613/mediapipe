import mediapipe as mp
import cv2
import pyautogui


#画像上の各骨格情報を取得する関数
def get_keypoint(results, height, width):

	#results.multi_hand_landmarks[0] 最初に検出できた手
	#results.multi_hand_landmarks[1] 片方の手

	#人差し指
	index_finger_x = int(results.multi_hand_landmarks[0].landmark[8].x * width)
	index_finger_y = int(results.multi_hand_landmarks[0].landmark[8].y * height)
	index_finger_xy = [index_finger_x, index_finger_y]

	return index_finger_xy


#骨格情報を描画する処理
def draw_keypoint(image, RADIUS, CLR_KP, index_finger_xy):

	cv2.circle(image, (index_finger_xy[0], index_finger_xy[1]), RADIUS, CLR_KP, THICKNESS)
	return image


if __name__ == "__main__":

	#縁の半径を5
	RADIUS = 5
	#線の太さ2
	THICKNESS = 2
	#座標の色
	CLR_KP = (0, 0, 255)

	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # カメラ画像の横幅を1280に設定
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # カメラ画像の縦幅を720に設定

	mp_hands = mp.solutions.hands

	#描画
	#結果を描画するソリューションを提供
	mp_drawing = mp.solutions.drawing_utils

	#描画の仕方の設定
	mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=2,  color=(0,255,0))
	mark_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0,0,255))


	with mp_hands.Hands(
			max_num_hands=1,
			min_detection_confidence=0.5,
			static_image_mode=False) as hands_detection:

			# カメラが開いている場合
			while cap.isOpened():
				success , img = cap.read()

				if not success:
					continue
				
				#左右反転
				image = cv2.flip(img,1)
				rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

				height = rgb_image.shape[0]
				width = rgb_image.shape[1]

				results = hands_detection.process(rgb_image)

				if not results.multi_hand_landmarks:
					print('not results')
				else:

					index_finger_xy = get_keypoint(results, height, width)

					#print(results.multi_hand_landmarks[0].landmark[8])
					#print(index_finger_xy)
					pyautogui.moveTo(index_finger_xy[0],index_finger_xy[1])

					image = draw_keypoint(image, RADIUS, CLR_KP, index_finger_xy)

				cv2.imshow('play movie', image)

				if cv2.waitKey(5) & 0xFF == 27:
					break

	cap.release()
	cv2.destroyAllWindows()