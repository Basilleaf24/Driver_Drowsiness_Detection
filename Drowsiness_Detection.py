from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import numpy as np
import os

# --- OPTIONAL: Enable if using Raspberry Pi ---
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    LIGHT_PIN = 18   # Example GPIO pin for cabin light
    GPIO.setup(LIGHT_PIN, GPIO.OUT)
    use_gpio = True
except ImportError:
    print("[INFO] RPi.GPIO not available - using software brightness enhancement.")
    use_gpio = False

# --- Initialize mixer ---
mixer.init()
mixer.music.load("music.wav")

# --- FUNCTIONS ---

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # 51, 59
    B = distance.euclidean(mouth[4], mouth[8])   # 53, 57
    C = distance.euclidean(mouth[0], mouth[6])   # 49, 55
    return (A + B) / (2.0 * C)

# --- PARAMETERS ---
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
MOUTH_AR_THRESH = 0.75
MOUTH_AR_CONSEC_FRAMES = 15
LIGHT_THRESH = 50       # Brightness threshold for dark cabin
BRIGHTNESS_GAIN = 40    # Software brightness boost amount

eye_counter = 0
yawn_counter = 0

# --- Initialize Dlib ---
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# --- MODE SELECTION ---
MODE = "video"   # Change to "images" to test on image set

# ===============================
# VIDEO MODE (REAL-TIME DETECTION)
# ===============================
if MODE == "video":
	# --- Video Capture ---
	cap = cv2.VideoCapture(0)

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		# Compute average brightness
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		brightness = np.mean(gray)

		# --- Cabin lighting control ---
		if brightness < LIGHT_THRESH:
			if use_gpio:
				GPIO.output(LIGHT_PIN, GPIO.HIGH)
			else:
				# Software compensation
				frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=BRIGHTNESS_GAIN)
			cv2.putText(frame, "Low Light", (10, 430),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
		else:
			if use_gpio:
				GPIO.output(LIGHT_PIN, GPIO.LOW)
			cv2.putText(frame, "Bright Light", (10, 430),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

		# --- Drowsiness and Yawn Detection ---
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		subjects = detect(gray, 0)

		for subject in subjects:
			shape = predict(gray, subject)
			shape = face_utils.shape_to_np(shape)

			# Eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0

			# Mouth
			mouth = shape[mStart:mEnd]
			mar = mouth_aspect_ratio(mouth)

			# Draw facial features
			cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 0), 1)

			# --- Eye (Drowsiness) ---
			if ear < EYE_AR_THRESH:
				eye_counter += 1
				if eye_counter >= EYE_AR_CONSEC_FRAMES:
					cv2.putText(frame, "ALERT! DROWSINESS DETECTED!", (10, 30),
								cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
					mixer.music.play()
			else:
				eye_counter = 0

			# --- Mouth (Yawn) ---
			if mar > MOUTH_AR_THRESH:
				yawn_counter += 1
				if yawn_counter >= MOUTH_AR_CONSEC_FRAMES:
					cv2.putText(frame, "ALERT! DROWSINESS DETECTED!", (10, 60),
								cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
					mixer.music.play()
			else:
				yawn_counter = 0

			# Display EAR and MAR
			cv2.putText(frame, f"EAR: {ear:.2f}", (320, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
			cv2.putText(frame, f"MAR: {mar:.2f}", (320, 50),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

		cv2.imshow("Drowsiness Monitoring System", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

	# --- Cleanup ---
	cap.release()
	cv2.destroyAllWindows()
	if use_gpio:
		GPIO.output(LIGHT_PIN, GPIO.LOW)
		GPIO.cleanup()
            
# ===============================
# IMAGE MODE (TESTING ON IMAGES)
# ===============================
elif MODE == "images":
    test_folder = "test_images"  # folder containing test images
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for file in image_files:
        img_path = os.path.join(test_folder, file)
        frame = cv2.imread(img_path)
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)

            # Draw landmarks
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 0), 1)

            # Label detection result
            status_text = ""
            if ear < EYE_AR_THRESH:
                status_text += "DROWSY "
            if mar > MOUTH_AR_THRESH:
                status_text += "YAWNING"

            if status_text == "":
                status_text = "Normal"

            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, status_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if "DROWSY" in status_text or "YAWNING" in status_text else (0, 255, 0), 2)

        cv2.imshow(file, frame)
        cv2.waitKey(0)
        cv2.destroyWindow(file)

    cv2.destroyAllWindows()