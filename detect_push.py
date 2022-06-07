
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

# 推流地址
rtmpUrl = "rtmp://qn.live-send.acg.tv/live-qn/?streamname=live_68783796_70316579&key=3d5eea1bd6edfd59c16b9bb9416f9a62&schedule=rtmp"
# rtmpUrl = "rtmp://127.0.0.1:1935/live"
# rtmpUrl = "rtmp://202.115.17.6:8002/live/test3"
# rtmpUrl = "rtmp://202.115.17.6:8002/live/test2"
# 是视频还是视频流,视频需要进行循环播放
video_path = 0
#video_path = r"rtsp://admin:cs237239@192.168.191.1:554/h265/ch1/main/av_stream"
# video_path = r"videos/3_23_1_s.mp4"
# 摄像头是否需要转置
transpose = False

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps:{} width:{} height:{}".format(fps, width, height))
# 设置FPS
cap.set(cv2.CAP_PROP_FPS,15)
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
           '-i', 'car.png',
           '-filter_complex','overlay',
           # '-r', str(fps),
           '-b:v ','800k',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           # '-rtmp_buffer',str(0),
           rtmpUrl]
           
p = sp.Popen(command, stdin=sp.PIPE)

def frame_put(frame_q,cap_path,push=True):

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
        if cap_path ==0 & transpose:    
            # 垂直翻转
            frame = cv2.flip(frame,0)
            # 水平翻转
            frame = cv2.flip(frame,1)
        # 推流
        if push:
            # ~ image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # ~ p.stdin.write(image.tobytes())
            p.stdin.write(frame.tobytes())
            
        frame_q.put(frame)
        frame_q.get() if frame_q.qsize() > 2 else time.sleep(0.03)
        # print("get")

    cap.release()
            
def notice(index_q):
    engine = pyttsx3.init()
    engine.setProperty('voice', 'zh')
    engine.setProperty('rate', 200)
    # labels = ["正常", "未定义", "无人", "分心", "抽烟", "使用手机", "喝水", "抓痒", "拿东西"]
    labels = ["正常", "未定义", "无人", "请保持注意力", "请不要在车上吸烟", "请尽快完成手机操作", "喝水时注意前方动态", "请保持双手扶住方向盘", "请不要在开车时回头拿东西"]
    pre_index = 0
    while True:

        cur_index = index_q.get()
        # print(pre_index,cur_index)
        if pre_index!=cur_index & cur_index>0:
            pre_index = cur_index
            engine.say(labels[cur_index])
            engine.runAndWait()
        
def predict_onnx_notice(frame_q,index_q,video=False):

    # onnx runtime
    onnx_path = r"weights/ghostnet_my_nv_05_acc=97.4856.onnx"
    onnx_session = onnxruntime.InferenceSession(onnx_path,None)
    
    labels = ["正常", "未定义", "无人", "分心", "抽烟", "使用手机", "喝水", "抓痒", "拿东西"]
    while True:
        # 没有项目自动阻塞
        frame = frame_q.get()
        start = time.time()
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(frame2, (224, 224))
        # print("resize time:", time.time()-start)
        image = np.float32(image) / 255.0
        image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
        image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
        
        image = image.transpose(2, 0, 1) # 转换轴，pytorch为channel first
        image = image.reshape(1, 3, 224, 224) # barch,channel,height,weight
        inputs = {onnx_session.get_inputs()[0].name: image}
        probs = onnx_session.run(None, inputs)
        index = np.argmax(probs)
        # index =0
        print(labels[index])
        predict_time = time.time() - start
        print("predict time:", predict_time)
        index_q.put(index)
        if index_q.qsize() > 1:
            index_q.get()

def detect_video_onnx_multi_t_notice_image(cap_path):

    frame_q =queue.Queue(maxsize=2)
    index_q = queue.Queue(maxsize=2)
    
    put_t = threading.Thread(target=frame_put, args=(frame_q, cap_path))
    predict_t = threading.Thread(target=predict_onnx_notice, args=(frame_q,index_q))
    notice_t = threading.Thread(target=notice, args=(index_q,))
    
    put_t.setDaemon(True)
    #predict_t.setDaemon(True)
    notice_t.setDaemon(True)
    # 启动进程
    put_t.start()
    predict_t.start()
    notice_t.start()
    predict_t.join()
            
if __name__ == '__main__':
    start = time.time()
    detect_video_onnx_multi_t_notice_image(video_path)
    end = time.time()
    total = end -start
    print("total time:{}".format(total))
