import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands


#描画
#結果を描画するソリューションを提供
mp_drawing = mp.solutions.drawing_utils

#描画の仕方の設定
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=2,  color=(0,255,0))
mark_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0,0,255))

img_path = 'hand.jpg'

#mediapipeの手検出の宣言部分

#検出精度 min_detection_confidence(0が最も低い)
with mp_hands.Hands(
		max_num_hands=2,
		min_detection_confidence=0.5,
		static_image_mode=True) as hands_detection:

	image = cv2.imread(img_path)
	image = cv2.resize(image, dsize=None, fx=0.3, fy=0.3)
	rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	height = rgb_image.shape[0]
	width = rgb_image.shape[1]

	results = hands_detection.process(rgb_image)
	annotated_image = image.copy()

	#一つの手に対して、ランドマークは22個
	for hand_landmarks in results.multi_hand_landmarks:
		# for id, lm in enumerate(hand_landmarks.landmark):
		# 	print(id, lm)

		#オリジナル画像 , 手検出 描画
		mp_drawing.draw_landmarks(
			image=annotated_image,
			landmark_list = hand_landmarks,
			connections = mp_hands.HAND_CONNECTIONS,
			landmark_drawing_spec = mark_drawing_spec,
			connection_drawing_spec = mesh_drawing_spec
			)

		cv2.imwrite('result.jpg', annotated_image)
