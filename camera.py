import cv2

camera = cv2.VideoCapture('/dev/video2')


while True:
    _, observation = camera.read()
    cv2.imshow('Arm', cv2.cvtColor(observation, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)