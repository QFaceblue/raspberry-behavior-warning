#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import numpy as np
import time
import cv2

yolo = YOLO()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
        # print(image.size)# weight,height
        # print(np.asarray(image).shape) # height,weight,channel
    except:
        print('Open Error! Try again!')
        continue
    else:
        # ~ start = time.time()
        # ~ r_image = yolo.detect_image(image)
        # ~ predect_time = time.time() - start
        # ~ print("predict time:", predect_time)
        # r_image.show()

        # ONNX
        start = time.time()
        r_image = yolo.detect_image_onnx(image)
        predect_time = time.time() - start
        print("predict time:", predect_time)
        # ~ # r_image.show()
        
        img = np.asarray(r_image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("wheel", img)
        cv2.waitKey (0)
        
        # start = time.time()
        # label,conf,boxes = yolo.detect_img(image)
        # predect_time = time.time() - start
        # print("predict time:", predect_time)
        # print(label,conf,boxes)
        # get boxes
        
        # ~ start = time.time()
        # ~ boxes,conf,label = yolo.get_boxes_onnx(image)
        # ~ print(boxes,conf,label)
        # ~ predect_time = time.time() - start
        # ~ print("predict time:", predect_time)

