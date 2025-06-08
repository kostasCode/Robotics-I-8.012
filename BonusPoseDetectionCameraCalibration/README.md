# "Perspective projection is easy, the hard part is understanding it"

In this project we explore perspective projection by trying to draw frames using our hands ,(3d points in space) using MediaPipe an open-source framework developed by Google for building pipelines to process video, audio, and other multimedia types.

A Convolutional Neural Network (CNN) for hand point detection is designed to find key landmarks (like fingertips and joints) from images.

![image](https://github.com/user-attachments/assets/0f798e5e-30ea-4f19-bfc4-8ffeffb733de)

By sourcing 3d points of intrest we project and tranfrom them to 2d pixel space and then constructing coordinate frames with the following method:

>[!NOTE]
> The video was made using manim (3blue1brown python library)

https://github.com/user-attachments/assets/650a4b61-011d-49f1-b729-f656443bfa83

> [!TIP]
>Camera calibration is a crucial step towards achieving accurate results. By visiting [calib.io](https://calib.io/pages/camera-calibration-pattern-generator) and then running [calibration.py](https://github.com/kostasCode/Robotics-I-8.012/blob/main/BonusPoseDetectionCameraCalibration/calibration.py) using a flat monitor you can find the intristic parameters of your camera and insert them into your main program ([MiniMachineVisionProject.py](https://github.com/kostasCode/Robotics-I-8.012/blob/main/BonusPoseDetectionCameraCalibration/MiniMachineVisionProject.py))

# Results of the project (camera turned off)

![image](https://github.com/user-attachments/assets/48f209e7-7b2a-4355-aa42-8b3135f52345)

Frame C is Closer to the camera and appears bigger, 
Frame F is Farther away from the camera and appears smaller.

>[!NOTE]
>Small note: If your camera is fliped just remove line 179 -> frame = cv2.flip(frame, 1)
