from ultralytics import YOLO

import numpy as np

import cv2 as cv

font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 1  # Size of the font
color = (0, 0, 0)  # Black color (BGR)
position = (10, 30)
thickness = 2
if __name__ == '__main__':
    camera = cv.VideoCapture(0)
    model = YOLO("./runs\\classify\\train\\weights\\best.pt")

    while True:
        ret , image = camera.read()
        if cv.waitKey(46) & 0XFF == ord('q'):
            break
        result = model(image)
        prob = result[0].probs.data.tolist()
        how = result[0].names[np.argmax(prob)]
        if max(result[0].probs.data.tolist()) < 0.8:
            cv.putText(image, "none", position, font, font_scale, color, thickness)
        else:
            cv.putText(image, how, position, font, font_scale, color, thickness)
        cv.imshow('frame', result[0].orig_img)
    camera.release()
    cv.destroyAllWindows()
