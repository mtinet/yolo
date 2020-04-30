from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()

camera = cv2.VideoCapture(0)

#ip캠을 사용하는 경우 아래 코드를 활성화시킴
#camera = cv2.VideoCapture("http://192.168.43.1:8080/video")

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(camera_input=camera,
                                output_file_path=os.path.join(execution_path, "camera_detected_1")
                                , frames_per_second=29, log_progress=True)
print(video_path)
