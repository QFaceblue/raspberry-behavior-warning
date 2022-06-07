import subprocess as sp
import cv2 as cv
import time

# ffmpeg -i drive.avi  -c:v libx264-f flv rtmp://qn.live-send.acg.tv/live-qn/?streamname=live_68783796_70316579&key=3d5eea1bd6edfd59c16b9bb9416f9a62&schedule=rtmp
# raspivid -o -| ffmpeg -i -  -f flv rtmp://qn.live-send.acg.tv/live-qn/?streamname=live_68783796_70316579&key=3d5eea1bd6edfd59c16b9bb9416f9a62&schedule=rtmp
##rtmpUrl = "rtmp://txy.live-send.acg.tv/live-txy/?streamname=live_68783796_70316579&key=3d5eea1bd6edfd59c16b9bb9416f9a62"
# rtmpUrl = "rtmp://qn.live-send.acg.tv/live-qn/?streamname=live_68783796_70316579&key=3d5eea1bd6edfd59c16b9bb9416f9a62&schedule=rtmp"
# rtmpUrl = "rtmp://127.0.0.1:1935/live"
# rtmpUrl = "rtmp://202.115.17.6:8002/live/test3"
# rtmpUrl = "rtmp://202.115.17.6:8002/live/test2"
rtmpUrl = "rtmp://202.115.17.6:40000/http_flv/test_raw"
rtmpUrl = "rtmp://192.168.2.222:50000/http_flv/test"
cap_path = 0
# cap_path ="./video/demo.avi"
cap = cv.VideoCapture(cap_path)

# Get video information
fps = int(cap.get(cv.CAP_PROP_FPS))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# ~ cap.set(cv.CAP_PROP_FPS,15)
# ~ fps = int(cap.get(cv.CAP_PROP_FPS))
print("fps:{} width:{} height:{}".format(fps, width, height))
# ffmpeg command
# 文件名为 -，则从标准输入设备读取数据。
command = ['ffmpeg',
           '-y',
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
# ~ command = ['ffmpeg',
        # ~ '-y',
        # ~ '-v','quiet', # 不输出推流信息
        # ~ '-f', 'rawvideo',
        # ~ '-vcodec','rawvideo',
        # ~ '-pix_fmt', 'bgr24',
        # ~ '-s', "{}x{}".format(width, height),
        # ~ #'-r', str(fps),
        # ~ '-i', '-',
        # ~ '-c:v', 'libx264',
        # ~ '-pix_fmt', 'yuv420p',
        # ~ '-preset', 'ultrafast',
        # ~ '-f', 'flv', 
        # ~ #'-rtmp_buffer',str(0),
        # ~ rtmpUrl]
# 管道配置
p = sp.Popen(command, stdin=sp.PIPE)

# 是否绘制帧信息
draw = False
# 摄像头是否需要转置
transpose = False
if draw:
    accum_time = 0
    curr_fps = 0
    true_fps = "FPS: ??"
    prev_time = time.time()
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        if cap_path != 0:
                print("replay the video")
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                continue
        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    if cap_path == 0 and transpose:    
            # 垂直翻转
            frame = cv.flip(frame,0)
            # 水平翻转
            frame = cv.flip(frame,1)
    if draw:
        curr_time = time.time()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            true_fps = "FPS:" + str(curr_fps)
            curr_fps = 0
        # t =time.asctime(time.localtime(time.time()))
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        text = "{} time:{}".format(true_fps, t)

        cv.putText(frame, text=text, org=(10, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=3)
    cv.imshow("front camera", frame)
    # print(cap.get(cv.CAP_PROP_POS_FRAMES))
    # time.sleep(1/fps)
    time.sleep(0.015)
    # write to pipe
    p.stdin.write(frame.tobytes())
    # p.stdin.write(frame.tostring())
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    

# def push_rtmp(pipe,video_path,rtmp_url):
#     return
