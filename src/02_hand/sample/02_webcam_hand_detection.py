import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands


#描画
#結果を描画するソリューションを提供
mp_drawing = mp.solutions.drawing_utils

#描画の仕方の設定
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=2,  color=(0,255,0))
mark_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0,0,255))

cap_file = cv2.VideoCapture(0)

with mp_hands.Hands(
		max_num_hands=2,
		min_detection_confidence=0.5,
		static_image_mode=False) as hands_detection:

		# カメラが開いている場合
		while cap_file.isOpened():
			success , img = cap_file.read()
			
			if not success:
				continue
			
			#左右反転
			image = cv2.flip(img,1)
			rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			results = hands_detection.process(rgb_image)

			#手の骨格を取得できた場合のみ、webカメラに描画する
			if results.multi_hand_landmarks:

				#一つの手に対して、ランドマークは22個
				for hand_landmarks in results.multi_hand_landmarks:
					# for id, lm in enumerate(hand_landmarks.landmark):
					# 	print(id, lm)

					#オリジナル画像 , 手検出 描画
					mp_drawing.draw_landmarks(
						image=image,
						landmark_list = hand_landmarks,
						connections = mp_hands.HAND_CONNECTIONS,
						landmark_drawing_spec = mark_drawing_spec,
						connection_drawing_spec = mesh_drawing_spec
						)

			cv2.imshow('play movie', image)

			if cv2.waitKey(5) & 0xFF == 27:
				break

cap_file.release()
cv2.destroyAllWindows()