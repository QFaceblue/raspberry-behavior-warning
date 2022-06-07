import subprocess as sp
import cv2 as cv
import time
#rtmpUrl = "rtmp://txy.live-send.acg.tv/live-txy/?streamname=live_68783796_70316579&key=3d5eea1bd6edfd59c16b9bb9416f9a62"
#rtmpUrl = "rtmp://202.115.17.6:40000/http_flv/test"
rtmpUrl = "rtmp://192.168.2.222:50000/http_flv/test_raw"
camera_path = 0
cap = cv.VideoCapture(camera_path)

# Get video information
fps = int(cap.get(cv.CAP_PROP_FPS))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print("fps:{} width:{} height:{}".format(fps,width,height))
# ffmpeg command
command = ['ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', "{}x{}".format(width, height),
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-f', 'flv', 
        '-rtmp_buffer',str(0),
        rtmpUrl]

# 管道配置
p = sp.Popen(command, stdin=sp.PIPE)

# read webcamera

accum_time = 0
curr_fps = 0
fps = "FPS: ??"
prev_time = time.time()
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("Opening camera is failed")
        break

    # process frame
    # your code
    # process frame
    # 垂直翻转
    # ~ frame = cv.flip(frame,0)
    curr_time = time.time()
    exec_time = curr_time - prev_time
    prev_time = curr_time
    accum_time = accum_time + exec_time
    curr_fps = curr_fps + 1
    if accum_time > 1:
        accum_time = accum_time - 1
        fps = "FPS:" + str(curr_fps)
        curr_fps = 0
    #t =time.asctime(time.localtime(time.time()))
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    text = "{} time:{}".format(fps, t)
    
    cv.putText(frame, text=text, org=(10, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=3)
    cv.imshow("lab", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
            break
            
    # write to pipe
    p.stdin.write(frame.tobytes())
    # p.stdin.write(frame.tostring())
