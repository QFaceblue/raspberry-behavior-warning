from socket import *
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
import os
import sys
import serial
from yolo import YOLO
import random
import json

with open("config/config.json", "r", encoding='utf-8') as f:
    config = json.load(f)
# 权重地址
weight = config["weight"]
# 行为数目
class_num = config["class_num"]
labels = config["labels"]
notices = config["notices"]
# 推流地址
#rtmpUrl = "rtmp://202.115.17.6:40000/http_flv/test"
rtmpUrl = config["rtmpUrl"]
# 是视频还是视频流,视频需要进行循环播放
# ~ video_path = 0
# ~ video_path = r"videos/xw.mp4"
video_path = config["video_path"]
# 摄像头是否需要转置
# ~ transpose = False
transpose = config["transpose"]
# 是否推流
# ~ push_rtmp = False
push_rtmp = config["push_rtmp"]
# 是否推检测流
# ~ push_detect = False
push_detect = config["push_detect"]
# 是否记录危险驾驶行为
record = config["record"]
# 置信度
score = config["score"]
push_detect = config["push_detect"]
# 是否使用vpu 需提前配置环境变量 source /opt/intel/openvino/bin/setupvars.sh
# ~ use_vpu = False
use_vpu = config["use_vpu"]
# 是否上传gps数据
# ~ gps = False
gps = config["gps"]

# ~ username="user1"
# ~ # 密码
# ~ password="password1"
username = config["username"]
password = config["password"]
# ~ #random
# ~ randint = random.randint(1,90)
# ~ # 用户名
# ~ username="user"+ str(randint)
# ~ # 密码
# ~ password="password"+ str(randint)
print(username,password)

# ~ reconnect_time = 1800
# ~ server_ip = "202.115.17.6"
# ~ cmd_port = 55530
# ~ file_port = 55531
# ~ buffer_size = 1024
reconnect_time = config["reconnect_time"]
server_ip = config["server_ip"]
cmd_port = config["cmd_port"]
file_port = config["file_port"]
buffer_size = config["buffer_size"]
# 速度记录
last_v = 0.0
current_v = 0.0
# gps record
gps_record = ""

# 通过该参数控制进程结束
stop_notice = False
# wheel box 
box = None
cap = cv2.VideoCapture(video_path)
# ~ try:
    # ~ cap = cv2.VideoCapture(video_path)
# ~ except Exception as e:
    # ~ print("can't found camera!")
    # ~ sys.exit(1)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps:{} width:{} height:{}".format(fps, width, height))
# 设置FPS
# ~ cap.set(cv2.CAP_PROP_FPS,15)
# ~ fps = int(cap.get(cv2.CAP_PROP_FPS))
# 设置图片大小
# ~ cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
# ~ cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
# ~ cap.set(cv2.CAP_PROP_FRAME_WIDTH,224)
# ~ cap.set(cv2.CAP_PROP_FRAME_HEIGHT,224)
# ~ width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# ~ height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# ~ print("fps:{} width:{} height:{}".format(fps, width, height))

# 管道配置
command = ['ffmpeg',
           '-y',
           '-v','quiet', # 不输出推流信息
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(width, height),
           # '-r', str(fps),
           '-i', '-',
           # '-i', 'car.png',
           # '-filter_complex','overlay',
           # '-r', str(fps),
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           # '-rtmp_buffer',str(0),
           rtmpUrl]
           
p = sp.Popen(command, stdin=sp.PIPE)

# 音频配置
engine = pyttsx3.init()
engine.setProperty('voice', 'zh')
engine.setProperty('rate', 200)

def login_val(socket, user, pwd):
    info = "{} {}".format(user, pwd)
    socket.send(info.encode())
    recved = socket.recv(1024)
    print(recved.decode())
    if recved == b"pass":
        return True
    else:
        return False
        
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
# 检测是否插入vpu
def detect_vpu2():
	for index,line in enumerate(os.popen("lsusb")):
		if "Intel Movidius MyriadX" in line:
			print("Found vpu!")	
			return True
	print("No vpu was detected!")	
	return False
# 检测是否连接北斗模组
def detect_beidou():
	for index,line in enumerate(os.popen("lsusb")):
		if "QinHeng Electronics HL-340 USB-Serial adapter" in line:
			print("Found beidou!")	
			return True
	print("No beidou was detected!")	
	return False    
    
def frame_put(frame_q,cap_path,push=False):
    print("{} rtmp start!".format(push))
    # 每个线程单独创建管道
    # ~ if push:
        # ~ p = sp.Popen(command, stdin=sp.PIPE)
    global push_rtmp
    while cap.isOpened():
        if push!=push_rtmp:
            print("{} push rtmp stop!".format(push))
            break
        return_value, frame = cap.read()
        if not return_value:
            if cap_path != 0:
                print("replay the video")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("Can't receive frame (stream end?). Exiting ...")
                break
        if cap_path ==0 and transpose:    
            # 垂直翻转
            frame = cv2.flip(frame,0)
            # 水平翻转
            frame = cv2.flip(frame,1)
        # 推流
        if push:
            global p
            global command
            print("push_rtmp")
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # p.stdin.write(image.tobytes())
            # push stream to bilibili
            try:
                p.stdin.write(image.tobytes())
            except Exception as e:
                print(e)
                print("create new pipe")
                p = sp.Popen(command, stdin=sp.PIPE)
            else:
                pass
        frame_q.put(frame)
        frame_q.get() if frame_q.qsize() > 2 else time.sleep(0.03)
        # ~ if frame_q.qsize() > 2:
            # ~ frame_q.get() 
        # ~ time.sleep(0.02)
        # print("get")

            
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

# ~ def notice(index_q,t=2):

    # ~ # engine = pyttsx3.init()
    # ~ # engine.setProperty('voice', 'zh')
    # ~ # engine.setProperty('rate', 200)
    # ~ global engine
    # ~ labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
    # ~ pre_index = 0
    # ~ time = 1
    # ~ # start
    # ~ print("say","检测开始，请保持正确的驾驶姿势！")
    # ~ engine.say("检测开始请保持正确的驾驶姿势！")
    # ~ engine.runAndWait()  
    # ~ global stop_notice
    # ~ while True:
        # ~ if stop_notice:
            # ~ break
        # ~ cur_index,prob = index_q.get()
        # ~ # print(pre_index,cur_index)
        # ~ if pre_index!=cur_index :
            # ~ pre_index = cur_index
            # ~ time = 1
        # ~ else:
            # ~ time = time + 1
            # ~ if time>=t and cur_index>0:
                # ~ # print("say",labels[cur_index])
                # ~ engine.say(labels[cur_index])
                # ~ engine.runAndWait()   
# 读取服务器发送的提示
def notice(index_q,notice_q,t1=2,t2=4):

    # engine = pyttsx3.init()
    # engine.setProperty('voice', 'zh')
    # engine.setProperty('rate', 200)
    global engine
    global notices
    # ~ labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
    pre_index = 0
    time = 1
    # start
    print("say","检测开始，请保持正确的驾驶姿势！")
    engine.say("检测开始请保持正确的驾驶姿势！")
    engine.runAndWait()  
    global stop_notice
    global score
    while True:
        if stop_notice:
            break
        # 读取服务器发送的提示
        if notice_q.qsize() > 0:
            notice = notice_q.get()
            engine.say(notice)
            engine.runAndWait() 
        cur_index,prob = index_q.get()
        # print(pre_index,cur_index)
        if pre_index!=cur_index :
            pre_index = cur_index
            time = 1
        else:
            time = time + 1
            if time>=t1 and time<t2 and  cur_index>0 and prob>score:
                # print("say",notices[cur_index])
                engine.say(notices[cur_index])
                engine.runAndWait()  
                
def predict_onnx_notice(frame_q,predict_q,index_q,record=False,crop=False):

    global gps_record
    global weight
    global labels
    global score
    # onnx runtime
    # ~ onnx_path = r"weights/mobilenetv2_1_12_23_acc=89.9061.onnx"
    # ~ onnx_path = r"weights/mobilenetv2_1_12_23_acc=91.6275.onnx"
    # ~ onnx_path = r"weights/mobilenetv2_1_crop_acc=90.9233.onnx"
    # ~ onnx_path = r"weights/mobilenetv2_224_acc=85.6154.onnx"
    onnx_path = weight
    if crop:
        frame = frame_q.get()
        boxes = get_boxes_frame(frame)
        if boxes is not None:
            # crop
            onnx_path = r"weights/mobilenetv2_1_crop_acc=94.8357.onnx"
            global box
            box = boxes[0]
            print(box)
    onnx_session = onnxruntime.InferenceSession(onnx_path,None)
    # ~ labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
    # record dangerous behaviouts of driver
    if record:
        img_savepath = "record/{}".format(time.strftime("%Y-%m-%d", time.localtime()))
        if not os.path.isdir(img_savepath):
            os.makedirs(img_savepath)
        index = 0
        pre_index = 0
        
    while True:
        # 没有项目自动阻塞
        time.sleep(0.15)
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
        # ~ print(labels[index],prob)
        
        if record:
            if index>0 and index!=pre_index and prob>score:
                img_name = "{}_{}_{}.png".format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),index,gps_record)
                img_path = os.path.join(img_savepath,img_name)
                cv2.imwrite(img_path,frame)
                pre_index = index
        predict_time = time.time() - start
        # ~ print("predict time:", predict_time)
        predict_q.put((index,prob))
        if predict_q.qsize() > 1:
            predict_q.get()
        index_q.put((index,prob))
        if index_q.qsize() > 1:
            index_q.get()

def predict_openvino_notice(frame_q,predict_q,index_q,record=False):

    xml_path = r"weights/mobilenetv2_1_my_224.xml"
    bin_path = r"weights/mobilenetv2_1_my_224.bin"
    xml_path = r"weights/mobilenetv2_1_12_23_acc=91.6275.xml"
    bin_path = r"weights/mobilenetv2_1_12_23_acc=91.6275.bin"
    net = cv2.dnn.readNet(xml_path,bin_path)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)    
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
            
def draw_image(frame_q,predict_q,title,push=False):
    global labels
    # ~ labels = ["正常", "侧视", "喝水", "吸烟", "操作中控", "玩手机", "侧身拿东西", "整理仪容", "接电话"]
    accum_time = 0
    curr_fps = 0
    detect_fps = 0
    fps = "FPS: ??"
    d_fps = "detect_FPS: ??"
    prev_time = timer()
    index = 0
    font = ImageFont.truetype(font='assets/simhei.ttf',size=int(30))#20
    
    print("{} push_detect start!".format(push))
    global push_detect
    global current_v
    global box
    global stop_notice
    prob =0.
    while True:
        if push != push_detect:
            break
        frame = frame_q.get()
        
        img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_PIL)
        if predict_q.qsize() > 0:
            index,prob = predict_q.get()
            detect_fps = detect_fps + 1
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS:" + str(curr_fps)
            d_fps = "detect_FPS:" + str(detect_fps)
            d_fps = "detect_FPS:" + str(2)
            curr_fps = 0
            detect_fps = 0
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # ~ text = "{}:{:.2f} 速度:{:.2f}km/h {} {} {}".format(labels[index],prob,current_v,t,fps,d_fps)
        # ~ draw.text((20, 50), text, fill=(255, 0, 0), font=font)
        text = "{}:{:.2f} 速度:{:.2f}km/h".format(labels[index],prob,current_v)
        draw.text((220, 50), text, fill=(255, 0, 0), font=font)
        text = "{} {} {}".format(t,fps,d_fps)
        draw.text((20, 20), text, fill=(255, 0, 0), font=font)
        frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
        # global box
        # print(box)
        if box is not None:
            top, left, bottom, right = box
            cv2.rectangle(frame,(int(left), int(top)),(int(right), int(bottom)),(0, 255, 0),2)
        
        # 推检测流
        add = True
        if push:
            global p
            global command
            print("push_detect")
            try:
                p.stdin.write(frame.tobytes())
                # 补帧
                if add :
                    p.stdin.write(frame.tobytes())
                add != add
            except Exception as e:
                print(e)
                print("create new pipe")
                p = sp.Popen(command, stdin=sp.PIPE)
            else:
                pass
                
        #cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_notice = True
            break

def detect_video_onnx_multi_t_notice_image(cap_path,title):

    frame_q =queue.Queue(maxsize=2)
    predict_q = queue.Queue(maxsize=2)
    index_q = queue.Queue(maxsize=2)
    
    put_t = threading.Thread(target=frame_put, args=(frame_q, cap_path))
    predict_t = threading.Thread(target=predict_onnx_notice, args=(frame_q, predict_q,index_q))
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
    cap.release()
    
def cmd(cap_path,title):
    global push_rtmp
    frame_q =queue.Queue(maxsize=2)
    predict_q = queue.Queue(maxsize=2)
    index_q = queue.Queue(maxsize=2)
    
    if push_rtmp:
        put_push_t = threading.Thread(target=frame_put, args=(frame_q, cap_path,True))
        put_push_t.setDaemon(True)
        put_push_t.start()
    else:
        put_t = threading.Thread(target=frame_put, args=(frame_q, cap_path))
        put_t.setDaemon(True)
        put_t.start()
    predict_t = threading.Thread(target=predict_onnx_notice, args=(frame_q, predict_q,index_q))
    draw_t = threading.Thread(target=draw_image, args=(frame_q,predict_q,title))
    notice_t = threading.Thread(target=notice, args=(index_q,))
    
    predict_t.setDaemon(True)
    notice_t.setDaemon(True)
    # 启动进程
    
    predict_t.start()
    draw_t.start()
    notice_t.start()
    
    while True:
        # 从终端读入用户输入的字符串
        cmd = input('>>> ')
        if cmd == 'exit':
            print(cmd)
            break
        if cmd =="start_push":
            if not push_rtmp:
                print("start_push")
                push_rtmp =True
                put_push_t = threading.Thread(target=frame_put, args=(frame_q, cap_path,True))
                put_push_t.setDaemon(True)
                put_push_t.start()
        elif cmd =="stop_push":
            if push_rtmp:
                print("stop_push")
                push_rtmp =False
                put_t = threading.Thread(target=frame_put, args=(frame_q, cap_path))
                put_t.setDaemon(True)
                put_t.start()
        else:
            print(cmd)
    
    # 等待绘制结束
    draw_t.join()
    cap.release()

def upload_record():
    global username
    global password
    global server_ip
    global serv
    # IP = '103.46.128.45'
    # SERVER_PORT = 36180
    # IP = '192.168.8.101'
    # ~ IP = '192.168.1.37'
    # ~ SERVER_PORT = 50000
    # ~ IP = '202.115.17.6'
    # ~ SERVER_PORT = 55531
    # ~ IP = '192.168.8.121'
    # ~ # IP = '192.168.8.122'
    # ~ SERVER_PORT = 55531
    # ~ BUFLEN = 1024

    IP = server_ip
    SERVER_PORT = file_port
    BUFLEN = buffer_size
    
    # 实例化一个socket对象，指明协议
    dataSocket = socket(AF_INET, SOCK_STREAM)

    # 连接服务端socket
    dataSocket.connect((IP, SERVER_PORT))
    # 用户验证
    if login_val(dataSocket, username, password):
        print("login！")
        pass
    else:
        print("refuse！")
        return
    info = "record"
    dataSocket.send(info.encode())
    # 等待接收服务端的消息
    recved = dataSocket.recv(BUFLEN)
    print(recved)
    if recved == b"yes":
        info = "ready"
        dataSocket.send(info.encode())
        print("ready")
        # 等待接收服务端的消息
        recved = dataSocket.recv(BUFLEN)
        base_path = "./record"
        if recved != b"":
            img_name = recved.decode("utf-8", "ignore")
            print(img_name)
            if img_name == "all":
                new_dirs = os.listdir(base_path)
            else:
                # print(img_name)
                file_name = os.path.splitext(img_name)
                # print(file_name)
                date = file_name[0].split("_")
                # print(date)
                dirs = os.listdir(base_path)
                # print(dirs)
                new_dirs = list(filter(lambda x: x >= date[0], dirs))
            new_dirs.sort()
            print(new_dirs)
            for nd in new_dirs:
                dir_path = os.path.join(base_path, nd)
                if img_name == "all":
                    new_imgs = os.listdir(dir_path)
                else:
                    img_name = img_name+"_9"
                    imgs = os.listdir(dir_path)
                    # print(imgs)
                    new_imgs = list(filter(lambda x: x > img_name, imgs))
                new_imgs.sort()
                print(new_imgs)
                if len(new_imgs) > 0:
                    dataSocket.send("dir".encode())
                    recved = dataSocket.recv(BUFLEN)
                    if recved ==b"yes":
                        dataSocket.send(nd.encode())
                        recved = dataSocket.recv(BUFLEN)
                        if recved ==b"start_dir":
                            for i in new_imgs:
                                dataSocket.send(i.encode())
                                recved = dataSocket.recv(BUFLEN)
                                if recved == b"start_img":
                                    img_path = os.path.join(dir_path, i)
                                    with open(img_path, "rb") as f:
                                        # for data in f:
                                        #     dataSocket.send(data)
                                        #     # print(data)
                                        while True:
                                            data = f.read(BUFLEN)
                                            if len(data) == 0:
                                                break
                                            dataSocket.send(data)
                                        print("doned")
                                    dataSocket.send("doned".encode())
                                    # 注意两端交互顺序，可以多次发送，避免多次发送数据何在一起被接收，需等待回复
                                    recved = dataSocket.recv(BUFLEN)
                                    if recved == b"doned":
                                        pass
                            dataSocket.send("dir_finished".encode())
                            # 注意两端交互顺序，可以多次发送，避免多次发送数据何在一起被接收，需等待回复
                            recved = dataSocket.recv(BUFLEN)
                            if recved ==b"dir_finished":
                                pass
                            print("dir_finished")


    dataSocket.close()

def get_v():
    if not detect_beidou():
        print("can't find beidou")
        return
    # 北斗
    ser = serial.Serial("/dev/ttyUSB0",9600)
    # global ser
    global gps
    global last_v
    global current_v
    global gps_record
    while True:
        if gps:
            break
        recv = ser.read(ser.inWaiting())
        if not recv =="":
            s = recv.decode()
            lines = s.split()
            for i in lines:
                if i.startswith('$GNRMC'):
                    
                    # print(i)
                    l = i.split(",") 
                    gps_record="_".join(l[3:8])
                    if len(l)<8:
                        continue                   
                    if l[7] =="":
                        v = 0.
                    else:
                        v = float(l[7])*1.852
                    # print("速度：{}km".format(v))
                    last_v = current_v
                    current_v = v
                    
            # break
        time.sleep(1)
        
def push_gps(save=False):
    
    global server_ip
    global cmd_port
    global file_port
    global buffer_size
    global gps
    global gps_record
    # global ser
    global last_v
    global current_v
    global username
    global password
    if not detect_beidou():
        print("can't find beidou")
        return

    # IP = '103.46.128.45'
    # SERVER_PORT = 36180
    # ~ IP = '192.168.1.37'
    # ~ # IP = '192.168.8.101'
    # ~ SERVER_PORT = 50000
    # ~ IP = '202.115.17.6'
    # ~ SERVER_PORT = 55531
    # ~ IP = '192.168.8.121'
    # ~ # IP = '192.168.8.122'
    # ~ SERVER_PORT = 55531
    # ~ BUFLEN = 1024
    
    IP = server_ip
    SERVER_PORT = file_port
    BUFLEN = buffer_size

    # 实例化一个socket对象，指明协议
    dataSocket = socket(AF_INET, SOCK_STREAM)

    # 连接服务端socket
    dataSocket.connect((IP, SERVER_PORT))
    # 用户验证
    if login_val(dataSocket, username, password):
        print("login！")
        pass
    else:
        print("refuse！")
        return
    info = "gps"
    dataSocket.send(info.encode())
    # 等待接收服务端的消息
    recved = dataSocket.recv(BUFLEN)
    print(recved)
    if recved == b"yes":
        ser = serial.Serial("/dev/ttyUSB0",9600)
        if save:
            log_path = "./logs"
            file_name = "gps_{}.txt".format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
            file_path = os.path.join(log_path, file_name)
            f= open(file_path,"w")
        while True:
            if not gps:
                break
            recv = ser.read(ser.inWaiting())
            if not recv =="":
                s = recv.decode()
                lines = s.split()
                # ~ print(lines)
                for i in lines:
                    if i.startswith('$GNRMC'):
                        dataSocket.send(i.encode())
                        if save:
                            f.write(i+"\n")
                            f.flush()
                        l = i.split(",") 
                        gps_record="_".join(l[3:8])
                        if len(l)<8:
                            continue                     
                        if l[7] =="":
                            v = 0.
                        else:
                            v = float(l[7])*1.852
                        # ~ print("速度：{}km".format(v))
                        last_v = current_v
                        current_v = v
            time.sleep(1)
    dataSocket.close()

# 主线程与服务器交互,不推检测流
def client(cap_path,title):

    global push_rtmp
    global gps
    global record
    global use_vpu
    global username
    global password
    global server_ip
    global cmd_port
    global file_port
    global buffer_size
    
    frame_q =queue.Queue(maxsize=2)
    predict_q = queue.Queue(maxsize=2)
    index_q = queue.Queue(maxsize=2)
    notice_q = queue.Queue(maxsize=2)
    
    if push_rtmp:
        put_push_t = threading.Thread(target=frame_put, name="put_push", args=(frame_q, cap_path,True))
        put_push_t.setDaemon(True)
        put_push_t.start()
    else:
        put_t = threading.Thread(target=frame_put, name="put", args=(frame_q, cap_path))
        put_t.setDaemon(True)
        put_t.start()
    
    if use_vpu:
        vpu = detect_vpu2()
        if vpu:
            predict_t = threading.Thread(target=predict_openvino_notice, name="predict", args=(frame_q, predict_q,index_q,record))
        else:
            predict_t = threading.Thread(target=predict_onnx_notice, name="predict", args=(frame_q, predict_q,index_q,record))
    else:
        predict_t = threading.Thread(target=predict_onnx_notice, name="predict", args=(frame_q, predict_q,index_q,record))
        
    draw_t = threading.Thread(target=draw_image, name="draw", args=(frame_q,predict_q,title))
    notice_t = threading.Thread(target=notice, name="notice", args=(index_q,notice_q))
    
    predict_t.setDaemon(True)
    ## 更改为等待提醒结束
    # notice_t.setDaemon(True)
    # 启动进程
    
    predict_t.start()
    draw_t.start()
    notice_t.start()
    # 启动速度获取
    get_v_t = threading.Thread(target=get_v, name="get_v",args=())
    get_v_t.setDaemon(True)
    get_v_t.start()
    # 连接服务器
    # IP = '192.168.8.101'
    # ~ SERVER_PORT = 50000
    # ~ IP = '192.168.1.37'
    # IP = '192.168.2.86'
    # ~ IP = '202.115.17.6'
    # ~ SERVER_PORT = 55530
    # ~ IP = '192.168.8.121'
    # ~ # IP = '192.168.8.122'
    # ~ SERVER_PORT = 55530
    # ~ # IP = '103.46.128.45'
    # ~ # SERVER_PORT = 36180
    # ~ BUFLEN = 1024
    IP = server_ip
    SERVER_PORT = cmd_port
    BUFLEN = buffer_size
    
    # 实例化一个socket对象，指明协议
    dataSocket = socket(AF_INET, SOCK_STREAM)
    # 连接服务端socket
    dataSocket.connect((IP, SERVER_PORT))
    # 用户验证
    if login_val(dataSocket, username, password):
        print("login！")
        pass
    else:
        print("refuse！")
        return
    info = "notice"
    dataSocket.send(info.encode())
    # 等待接收服务端的消息
    recved = dataSocket.recv(BUFLEN)
    if recved == b"yes":
        dataSocket.send("ready".encode())
        while True:
            # 显示运行线程
            # ~ for t in threading.enumerate():
                # ~ print(t.getName())
            # 从服务器获取指令
            cmd = dataSocket.recv(BUFLEN)
            cmd = cmd.decode("utf-8", "ignore")
            if cmd == 'exit':
                print(cmd)
                break
            if cmd =="start_push":
                if not push_rtmp:
                    print("start_push")
                    push_rtmp =True
                    put_push_t = threading.Thread(target=frame_put, name="put_push", args=(frame_q, cap_path,True))
                    put_push_t.setDaemon(True)
                    put_push_t.start()
            elif cmd =="stop_push":
                if push_rtmp:
                    print("stop_push")
                    push_rtmp =False
                    put_t = threading.Thread(target=frame_put, name="put", args=(frame_q, cap_path))
                    put_t.setDaemon(True)
                    put_t.start()
            elif cmd =="upload_record":
                print("upload driver violation record! ")
                upload_t = threading.Thread(target=upload_record, name="upload_record")
                upload_t.setDaemon(True)
                upload_t.start()
            elif cmd =="push_gps":
                gps = True
                print("push_gps")
                gps_t = threading.Thread(target=push_gps, name="push_gps",args=())
                gps_t.setDaemon(True)
                gps_t.start()
            elif cmd =="stop_gps":
                print("stop_gps")
                gps = False
                get_v_t = threading.Thread(target=get_v, name="get_v",args=())
                get_v_t.setDaemon(True)
                get_v_t.start()
            else:
                print(cmd)
    
    # 等待绘制结束
    # draw_t.join()
    # 等待提醒结束
    notice_t.join()
    cap.release()

def conncet(cap_path,frame_q,predict_q,index_q,notice_q,title):

    global push_rtmp
    global push_detect
    global gps
    global record
    global use_vpu
    global engine
    global username
    global password
    global server_ip
    global cmd_port
    global file_port
    global buffer_size
    global reconnect_time
    # 连接服务器
    
    
    # ~ IP = '202.115.17.6'
    # ~ IP = '192.168.8.101'
    # ~ IP = '192.168.1.37'
    # ~ SERVER_PORT = 50000
    # ~ IP = '202.115.17.6'
    # ~ SERVER_PORT = 55530
    # ~ IP = '192.168.8.121'
    # ~ # IP = '192.168.8.122'
    # ~ SERVER_PORT = 55530
    # ~ BUFLEN = 1024
    
    IP = server_ip
    SERVER_PORT = cmd_port
    BUFLEN = buffer_size
    # 实例化一个socket对象，指明协议
    print("start conncet! "+IP+":"+str(SERVER_PORT))
    try:
        dataSocket = socket(AF_INET, SOCK_STREAM)
        # 连接服务端socket
        dataSocket.connect((IP, SERVER_PORT))
        print("connect")
    except Exception as e:
        print(e)
        print("connect fail!")
        time.sleep(reconnect_time)
        conncet(cap_path,frame_q,predict_q,index_q,notice_q,title)
        return 
    print("conncet success!")
    # 用户验证
    if login_val(dataSocket, username, password):
        print("login！")
        pass
    else:
        print("refuse！")
        return
    info = "notice"
    dataSocket.send(info.encode())
    # 等待接收服务端的消息
    recved = dataSocket.recv(BUFLEN)
    if recved == b"yes":
        dataSocket.send("ready".encode())
        while True:
            # 显示运行线程
            # ~ for t in threading.enumerate():
                # ~ print(t.getName())
            # 从服务器获取指令
            cmd = dataSocket.recv(BUFLEN)
            cmd = cmd.decode("utf-8", "ignore")
            if cmd == 'exit':
                print(cmd)
                info = "exit"
                dataSocket.send(info.encode())
                break
            if cmd =="start_push":
                if not push_rtmp and not push_detect:
                    print("start_push")
                    push_rtmp =True
                    put_push_t = threading.Thread(target=frame_put, name="put_push", args=(frame_q, cap_path,True))
                    put_push_t.setDaemon(True)
                    put_push_t.start()
                    info = "start_push"
                    dataSocket.send(info.encode())
                else:
                    info = "There is already a push thread"
                    dataSocket.send(info.encode())
            elif cmd =="stop_push":
                if push_rtmp:
                    print("stop_push")
                    push_rtmp =False
                    put_t = threading.Thread(target=frame_put, name="put", args=(frame_q, cap_path))
                    put_t.setDaemon(True)
                    put_t.start()
                    info = "stop_push"
                    dataSocket.send(info.encode())
                else:
                    info = "reuse stop_push"
                    dataSocket.send(info.encode())
                    
            elif cmd =="push_detect":
                if not push_detect and not push_rtmp:
                    print("push_detect")
                    push_detect =True
                    draw_push_t = threading.Thread(target=draw_image, name="draw_push", args=(frame_q,predict_q,title,True))
                    draw_push_t.setDaemon(True)
                    draw_push_t.start()
                    info = "push_detect"
                    dataSocket.send(info.encode())
                else:
                    info = "There is already a push thread"
                    dataSocket.send(info.encode())
            elif cmd =="stop_push_detect":
                if push_detect:
                    print("stop_push_detect")
                    push_detect =False
                    draw_t = threading.Thread(target=draw_image, name="draw", args=(frame_q,predict_q,title))
                    draw_t.setDaemon(True)
                    draw_t.start()
                    info = "stop_push_detect"
                    dataSocket.send(info.encode())
                else:
                    info = "reuse stop_push_detect"
                    dataSocket.send(info.encode())
            elif cmd =="upload_record":
                print("upload driver violation record! ")
                upload_t = threading.Thread(target=upload_record, name="unload_record")
                upload_t.setDaemon(True)
                upload_t.start()
                info = "upload_record"
                dataSocket.send(info.encode())
            elif cmd =="push_gps":
                gps = True
                print("push_gps")
                gps_t = threading.Thread(target=push_gps, name="push_gps",args=())
                gps_t.setDaemon(True)
                gps_t.start()
                info = "push_gps"
                dataSocket.send(info.encode())
            elif cmd =="stop_gps":
                print("stop_gps")
                gps = False
                get_v_t = threading.Thread(target=get_v, name="get_v",args=())
                get_v_t.setDaemon(True)
                get_v_t.start()
                info = "stop_gps"
                dataSocket.send(info.encode())
            else:
                print(cmd)
                notice_q.put(cmd)
                dataSocket.send(info.encode())
    
    # 等待绘制结束
    # draw_t.join()
    cap.release()

def main(cap_path,title):

    global push_rtmp
    global push_detect
    global gps
    global record
    global use_vpu
    frame_q =queue.Queue(maxsize=2)
    predict_q = queue.Queue(maxsize=2)
    index_q = queue.Queue(maxsize=2)
    notice_q = queue.Queue(maxsize=2)
    
    if push_rtmp:
        put_push_t = threading.Thread(target=frame_put, name="put_push", args=(frame_q, cap_path,True))
        put_push_t.setDaemon(True)
        put_push_t.start()
    else:
        put_t = threading.Thread(target=frame_put, name="put", args=(frame_q, cap_path))
        put_t.setDaemon(True)
        put_t.start()
    
    if push_detect and not push_rtmp:
        draw_push_t = threading.Thread(target=draw_image, name="draw_push", args=(frame_q,predict_q,title,True))
        draw_push_t.setDaemon(True)
        draw_push_t.start()
    else:
        draw_t = threading.Thread(target=draw_image, name="draw", args=(frame_q,predict_q,title))
        draw_t.setDaemon(True)
        draw_t.start()
    
    # 启动检测
    if use_vpu:
        vpu = detect_vpu2()
        if vpu:
            predict_t = threading.Thread(target=predict_openvino_notice, name="predict", args=(frame_q, predict_q,index_q,record))
        else:
            predict_t = threading.Thread(target=predict_onnx_notice, name="predict", args=(frame_q, predict_q,index_q,record))
    else:
        predict_t = threading.Thread(target=predict_onnx_notice, name="predict", args=(frame_q, predict_q,index_q,record))
    predict_t.setDaemon(True)
    predict_t.start()
    
    # 启动提示
    notice_t = threading.Thread(target=notice, name="notice", args=(index_q,notice_q))
    ## 更改为等待提醒结束
    # notice_t.setDaemon(True)
    notice_t.start()
    # 启动速度获取
    get_v_t = threading.Thread(target=get_v, name="get_v",args=())
    get_v_t.setDaemon(True)
    get_v_t.start()
    # 连接服务器
    connet_t = threading.Thread(target=conncet, name="connnet_t",args=(cap_path,frame_q,predict_q,index_q,notice_q,title))
    connet_t.setDaemon(True)
    connet_t.start()
    # 等待绘制结束
    # draw_t.join()
    # 更改为等待提醒结束
    # print(threading.enumerate())
    notice_t.join()
    cap.release()
    
if __name__ == '__main__':

    start = time.time()
    # detect_video_onnx_multi_t_notice_image(video_path, "driver")
    # cmd(video_path, "driver")
    # client(video_path, "driver")
    main(video_path, "driver")
    end = time.time()
    total = end -start
    print("total time:{}".format(total))
