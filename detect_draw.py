import cv2
import numpy as np
from timeit import default_timer as timer
import time
import multiprocessing as mp
import subprocess as sp
import threading
import queue
import onnxruntime
import pyttsx3
from PIL import Image,ImageFont, ImageDraw
import sys
import os 
from yolo import YOLO
# 是视频还是视频流,视频需要进行循环播放
video_path = 0
# video_path = r"rtsp://admin:cs237239@192.168.191.1:554/h265/ch1/main/av_stream"
video_path = r"videos/3_23_1_s.mp4"
# 摄像头是否需要转置
transpose = False
# 是否利用方向盘检测
box = None
# 类别数目
classnum = 9
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps:{} width:{} height:{}".format(fps, width, height))
# 设置FPS
# ~ cap.set(cv2.CAP_PROP_FPS,15)

def get_boxes_frame(frame):
    yolo = YOLO()
    image = Image.fromarray(np.uint8(frame))
    boxes,conf,label = yolo.get_boxes_onnx(image)
    return boxes
    
def softmax_np(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax
    
def frame_put(frame_q,cap_path,push=False):

    while cap.isOpened():
        return_value, frame = cap.read()
        if not return_value:
            if cap_path != 0:
                print("replay the video")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("Can't receive frame (stream end?). Exiting ...")
                break
        if video_path ==0 and transpose:    
            # 垂直翻转
            frame = cv2.flip(frame,0)
            # 水平翻转
            frame = cv2.flip(frame,1)

        frame_q.put(frame)
        frame_q.get() if frame_q.qsize() > 2 else time.sleep(0.03)
        # print("get")

    cap.release()
            
# ~ def notice(index_q):
    # ~ engine = pyttsx3.init()
    # ~ engine.setProperty('voice', 'zh')
    # ~ engine.setProperty('rate', 200)
    # ~ labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
    # ~ pre_index = 0
    # ~ while True:

        # ~ cur_index = index_q.get()
        # ~ # print(pre_index,cur_index)
        # ~ if pre_index!=cur_index & cur_index>0:
            # ~ pre_index = cur_index
            # ~ engine.say(labels[cur_index])
            # ~ engine.runAndWait()

def notice(index_q,t=2):
    # pass
    global classnum
    engine = pyttsx3.init()
    engine.setProperty('voice', 'zh')
    engine.setProperty('rate', 200)
    if classnum == 6:
        labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话"]
    elif classnum == 7:
        labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话","其他"]
    else:
        labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
    pre_index = 0
    time = 1
    # start
    print("say","检测开始，请保持正确的驾驶姿势！")
    engine.say("检测开始请保持正确的驾驶姿势！")
    engine.runAndWait()  
    while True:

        cur_index,prob = index_q.get()
        # print(pre_index,cur_index)
        if pre_index!=cur_index :
            pre_index = cur_index
            time = 1
        else:
            time = time + 1
            if time>=t and cur_index>0:
                print("say",labels[cur_index])
                engine.say(labels[cur_index])
                engine.runAndWait()      
                  
def predict_onnx_notice(frame_q,predict_q,index_q,record=False,crop=True):
    global classnum
    if classnum == 6:
        onnx_path = r"weights/mobilenetv2_1_c6_acc=95.1313.onnx"
    elif classnum == 7:
        onnx_path = r"weights/mobilenetv2_1_c6_acc=95.1313.onnx"
    else:
        onnx_path = r"weights/mobilenetv2_1_my_224.onnx"
        onnx_path = r"weights/mobilenetv2_1_my_acc=92.8962.onnx"
        onnx_path = r"weights/mobilenetv2_1_12_23_acc=89.9061.onnx"
        onnx_path = r"weights/mobilenetv2_1_12_23_acc=91.6275.onnx"
        onnx_path = r"weights/mobilenetv2_1_12_23_acc=88.2629..onnx"
    if crop:
        frame = frame_q.get()
        boxes = get_boxes_frame(frame)
        if boxes is not None:
            # crop
            if classnum == 6:
                onnx_path = r"weights/mobilenetv2_1_c6_acc=95.1313.onnx"
            else:
                onnx_path = r"weights/mobilenetv2_1_crop_acc=94.8357.onnx"
            global box
            box = boxes[0]
            print(box)
    onnx_session = onnxruntime.InferenceSession(onnx_path,None)
    if classnum == 6:
        labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话"]
    elif classnum == 7:
        labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话","其他"]
    else:
        labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
    # record dangerous behaviouts of driver
    if record:
        img_savepath = "record/{}".format(time.strftime("%Y-%m-%d", time.localtime()))
        if not os.path.isdir(img_savepath):
            os.makedirs(img_savepath)
        index = 0
        pre_index = 0
        
    while True:
        # 没有项目自动阻塞
        
        frame = frame_q.get()
        start = time.time()
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if box is not None:
            top, left, bottom, right = box
            frame2 = frame2[:,0:int(right)]
            
        image = cv2.resize(frame2, (224, 224))
        # image = cv2.resize(frame2, (160, 160))
        # print("resize time:", time.time()-start)
        image = np.float32(image) / 255.0
        image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
        image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
        
        image = image.transpose(2, 0, 1) # 转换轴，pytorch为channel first
        image = image.reshape(1, 3, 224, 224) # barch,channel,height,weight
        # image = image.reshape(1, 3, 160, 160) # barch,channel,height,weight
        inputs = {onnx_session.get_inputs()[0].name: image}
        # print("preprocess time:", time.time() -start)
        # 注意返回为三维数组（1,1,class_num)
        probs = onnx_session.run(None, inputs)
        index = np.argmax(probs)
        # print(probs)
        softmax_probs = softmax_np(np.array(probs))
        prob = softmax_probs.max()
        # print(index,prob)
        # index =0
        print(labels[index],prob)
        
        if record:
            if index>0 and index!=pre_index and prob>0.9:
                img_name = "{}_{}.png".format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),index)
                img_path = os.path.join(img_savepath,img_name)
                cv2.imwrite(img_path,frame)
                pre_index = index
        predict_time = time.time() - start
        print("predict time:", predict_time)
        predict_q.put((index,prob))
        if predict_q.qsize() > 1:
            predict_q.get()
        index_q.put((index,prob))
        if index_q.qsize() > 1:
            index_q.get()

def predict_onnx_notice_record(frame_q,predict_q,index_q,record=False):
    global classnum
    
    # onnx runtime
    if classnum == 6:
        onnx_path = r"weights/mobilenetv2_1_c6_acc=95.1313.onnx"
    else:
        onnx_path = r"weights/mobilenetv2_1_my_224.onnx"
        onnx_path = r"weights/mobilenetv2_1_my_acc=92.8962.onnx"
        onnx_path = r"weights/mobilenetv2_1_12_23_acc=89.9061.onnx"
        onnx_path = r"weights/mobilenetv2_1_12_23_acc=91.6275.onnx"
        onnx_path = r"weights/mobilenetv2_1_12_23_acc=88.2629..onnx"
        # onnx_path = r"weights/mnext_1_12_23_acc=92.1753.onnx"
        # onnx_path = r"weights/mobilenetv2_1_my_224c.onnx"
        # onnx_path = r"weights/mobilenetv2_1_my_160c.onnx"
        # onnx_path = r"weights/shufflenetv2_1.onnx"
        # onnx_path = r"weights/shufflenetv2_05_my.onnx"
    onnx_session = onnxruntime.InferenceSession(onnx_path,None)
    img_savepath = "record/{}".format(time.strftime("%Y-%m-%d", time.localtime()))
    if not os.path.isdir(img_savepath):
        os.makedirs(img_savepath)
    if classnum == 6:
        labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话"]
    elif classnum == 7:
        labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话","其他"]
    else:
        labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
    index = 0
    pre_index = 0
    while True:
        # 没有项目自动阻塞
        
        frame = frame_q.get()
        start = time.time()
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # crop crop_bbox = (0, 0, 1252, 1296) 2304*1296
        # ~ cw = int(1252/2304*640)
        # ~ ch = 360
        # ~ frame2 = frame2[0:ch,0:cw]
        image = cv2.resize(frame2, (224, 224))
        # image = cv2.resize(frame2, (160, 160))
        # print("resize time:", time.time()-start)
        image = np.float32(image) / 255.0
        image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
        image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
        
        image = image.transpose(2, 0, 1) # 转换轴，pytorch为channel first
        image = image.reshape(1, 3, 224, 224) # barch,channel,height,weight
        # image = image.reshape(1, 3, 160, 160) # barch,channel,height,weight
        inputs = {onnx_session.get_inputs()[0].name: image}
        # print("preprocess time:", time.time() -start)
        # 注意返回为三维数组（1,1,class_num)
        probs = onnx_session.run(None, inputs)
        index = np.argmax(probs)
        # print(probs)
        softmax_probs = softmax_np(np.array(probs))
        prob = softmax_probs.max()
        # print(index,prob)
        # index =0
        print(labels[index],prob)
        if index>0 and index!=pre_index and prob>0.9:
            img_name = "{}_{}.png".format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),index)
            img_path = os.path.join(img_savepath,img_name)
            cv2.imwrite(img_path,frame)
            pre_index = index

        predict_time = time.time() - start
        print("predict time:", predict_time)
        predict_q.put((index,prob))
        
        if predict_q.qsize() > 1:
            predict_q.get()
        index_q.put((index,prob))
        if index_q.qsize() > 1:
            index_q.get()
            
# source /opt/intel/openvino/bin/setupvars.sh
def predict_openvino_notice(frame_q,predict_q,index_q,record=False):
    # ~ global classnum 
    
    xml_path = r"weights/mobilenetv2_1_my_224.xml"
    bin_path = r"weights/mobilenetv2_1_my_224.bin"
    xml_path = r"weights/mobilenetv2_1_12_23_acc=91.6275.xml"
    bin_path = r"weights/mobilenetv2_1_12_23_acc=91.6275.bin"
    # xml_path = r"weights/mobilenetv2_1_12_23_acc=91.6275_FP16.xml"
    # bin_path = r"weights/mobilenetv2_1_12_23_acc=91.6275_FP16.bin"
    xml_path = r"weights/mnext_1_12_23_acc=92.1753.xml"
    bin_path = r"weights/mnext_1_12_23_acc=92.1753.bin"
    net = cv2.dnn.readNet(xml_path,bin_path)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)  
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  
    
    # ~ if classnum == 6:
        # ~ labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话"]
    # ~ elif classnum == 7:
        # ~ labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话","其他"]
    # ~ else:
        # ~ labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
    labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
    # record dangerous behaviouts of driver
    if record:
        img_savepath = "record/{}".format(time.strftime("%Y-%m-%d", time.localtime()))
        if not os.path.isdir(img_savepath):
            os.makedirs(img_savepath)
        index = 0
        pre_index = 0
    while True:
        # 没有项目自动阻塞
        frame = frame_q.get()
        start = time.time()
        # time.sleep(0.3)
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # crop crop_bbox = (0, 0, 1252, 1296) 2304*1296
        # ~ cw = int(1252/2304*640)
        # ~ ch = 360
        # ~ frame2 = frame2[0:ch,0:cw]
        image = cv2.resize(frame2, (224, 224))
        # image = cv2.resize(frame2, (160, 160))
        # print("resize time:", time.time()-start)
        image = np.float32(image) / 255.0
        image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
        image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
        
        # image = image.transpose(2, 0, 1) # 转换轴，pytorch为channel first
        # image = image.reshape(1, 3, 224, 224) # barch,channel,height,weight
        # image = image.reshape(1, 3, 160, 160) # barch,channel,height,weight
        # 注意返回为二维数组（1,class_num)
        print(image.shape)
        blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (0, 0, 0), False)
        net.setInput(blob)
        probs = net.forward()
        # np.argmax若不指定axis，则将数组铺平取值
        index = np.argmax(probs)
        softmax = softmax_np(probs)
        prob = softmax.max()
        # index =0
        print(labels[index],prob)
        if record:
            if index>0 and index!=pre_index and prob>0.9:
                img_name = "{}_{}.png".format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),index)
                img_path = os.path.join(img_savepath,img_name)
                cv2.imwrite(img_path,frame)
                pre_index = index
        predict_time = time.time() - start
        print("predict time:", predict_time)
        predict_q.put((index,prob))
        if predict_q.qsize() > 1:
            predict_q.get()
        index_q.put((index,prob))
        if index_q.qsize() > 1:
            index_q.get()
            
def draw_image(frame_q,predict_q,title):
    global classnum
    
    if classnum == 6:
        labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话"]
    elif classnum == 7:
        labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话","其他"]
    else:
        labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    index = 0
    prob = 0.
    font = ImageFont.truetype(font='assets/simhei.ttf',size=int(40))
    while True:
        frame = frame_q.get()
        
        img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_PIL)
        if predict_q.qsize() > 0:
            index,prob = predict_q.get()

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        # text = "label:{}   {}".format(labels[index], fps)
        text = "label:{} {:.2f} {}".format(labels[index], prob,fps) 
        draw.text((30, 20), text, fill=(255, 0, 0), font=font)
        frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
        global box
        # print(box)
        if box is not None:
            top, left, bottom, right = box
            cv2.rectangle(frame,(int(left), int(top)),(int(right), int(bottom)),(0, 255, 0),2)
        #cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def draw_image_save(frame_q,predict_q,title,save_path):
    # ~ global box
    global classnum
    
    if classnum == 6:
        labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话"]
    elif classnum == 7:
        labels = ["正常", "喝水", "吸烟", "操作中控", "玩手机", "接电话","其他"]
    else:
        labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    index = 0
    font = ImageFont.truetype(font='assets/simhei.ttf',size=int(40))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(save_path, fourcc, 20.0, (640, 480))
    while True:
        frame = frame_q.get()
        
        img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_PIL)
        if predict_q.qsize() > 0:
            index,prob = predict_q.get()

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        # text = "label:{}   {}".format(labels[index], fps)
        text = "label:{} {:.2f} {}".format(labels[index], prob,fps)
                
        draw.text((30, 20), text, fill=(255, 0, 0), font=font)
        frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
        
        # ~ print(box)
        # ~ if box is not None:
            # ~ top, left, bottom, right = box
            # ~ cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        #cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        out.write(frame)
        cv2.imshow(title, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
            
def detect_video_onnx_multi_t_notice_image(cap_path,title,record=False):

    frame_q =queue.Queue(maxsize=2)
    predict_q = queue.Queue(maxsize=2)
    index_q = queue.Queue(maxsize=2)
    
    put_t = threading.Thread(target=frame_put, args=(frame_q, cap_path))
    predict_t = threading.Thread(target=predict_onnx_notice, args=(frame_q, predict_q,index_q,record))
    draw_t = threading.Thread(target=draw_image, args=(frame_q,predict_q,title))
    notice_t = threading.Thread(target=notice, args=(index_q,))
    
    put_t.setDaemon(True)
    predict_t.setDaemon(True)
    notice_t.setDaemon(True)
    # 启动进程
    put_t.start()
    predict_t.start()
    draw_t.start()
    notice_t.start()
    # 等待绘制结束
    draw_t.join()

def detect_video_onnx_multi_t_notice_record(cap_path,title):

    frame_q =queue.Queue(maxsize=2)
    predict_q = queue.Queue(maxsize=2)
    index_q = queue.Queue(maxsize=2)
    
    put_t = threading.Thread(target=frame_put, args=(frame_q, cap_path))
    predict_t = threading.Thread(target=predict_onnx_notice_record, args=(frame_q, predict_q,index_q))
    draw_t = threading.Thread(target=draw_image, args=(frame_q,predict_q,title))
    notice_t = threading.Thread(target=notice, args=(index_q,))
    
    put_t.setDaemon(True)
    predict_t.setDaemon(True)
    notice_t.setDaemon(True)
    # 启动进程
    put_t.start()
    predict_t.start()
    draw_t.start()
    notice_t.start()
    # 等待绘制结束
    draw_t.join()
    
def detect_video_onnx_multi_t_notice_image_save(cap_path,title,save_path):

    frame_q =queue.Queue(maxsize=2)
    predict_q = queue.Queue(maxsize=2)
    index_q = queue.Queue(maxsize=2)
    
    put_t = threading.Thread(target=frame_put, args=(frame_q, cap_path))
    predict_t = threading.Thread(target=predict_onnx_notice, args=(frame_q, predict_q,index_q))
    draw_t = threading.Thread(target=draw_image_save, args=(frame_q,predict_q,title,save_path))
    notice_t = threading.Thread(target=notice, args=(index_q,))
    
    put_t.setDaemon(True)
    predict_t.setDaemon(True)
    notice_t.setDaemon(True)
    # 启动进程
    put_t.start()
    predict_t.start()
    draw_t.start()
    notice_t.start()
    # 等待绘制结束
    draw_t.join()
    
# source /opt/intel/openvino/bin/setupvars.sh
def detect_video_openvino_multi_t_notice_image(cap_path,title):

    frame_q =queue.Queue(maxsize=2)
    predict_q = queue.Queue(maxsize=2)
    index_q = queue.Queue(maxsize=2)
    
    put_t = threading.Thread(target=frame_put, args=(frame_q, cap_path))
    predict_t = threading.Thread(target=predict_openvino_notice, args=(frame_q, predict_q,index_q))
    draw_t = threading.Thread(target=draw_image, args=(frame_q,predict_q,title))
    notice_t = threading.Thread(target=notice, args=(index_q,))
    
    put_t.setDaemon(True)
    predict_t.setDaemon(True)
    notice_t.setDaemon(True)
    # 启动进程
    put_t.start()
    predict_t.start()
    draw_t.start()
    notice_t.start()
    # 等待绘制结束
    draw_t.join()

def get_temp():
    line = os.popen("cat /sys/class/thermal/thermal_zone0/temp").readline()
    temp = float(line)/1000.
    return temp
    
if __name__ == '__main__':
    start = time.time()
    s_temp = get_temp()
    if len(sys.argv)==1 :
        detect_video_onnx_multi_t_notice_image(video_path, "driver")
    elif sys.argv[1]=="save" :
        print("save video")
        save_path = "./videos/{}.avi".format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        detect_video_onnx_multi_t_notice_image_save(video_path, "save_video", save_path)
    elif sys.argv[1]=="record" :
        print("record")
        # detect_video_onnx_multi_t_notice_record(video_path, "record")
        detect_video_onnx_multi_t_notice_image(video_path, "driver")
    else:
        # source /opt/intel/openvino/bin/setupvars.sh
        print("use vpu")
        detect_video_openvino_multi_t_notice_image(video_path, "useVPU")
    end = time.time()
    total = end -start
    e_temp = get_temp()
    print("total time:{}".format(total))
    print("start temp={} end_temp={}".format(s_temp,e_temp))
