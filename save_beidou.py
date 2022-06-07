#  ===  推送gps端口数据客户端程序 client_serial.py ===
import serial #导入serial模块
import time
import os 
# 北斗
ser = serial.Serial("/dev/ttyUSB0",9600)
log_path = "./logs"
file_name = "gps_{}.txt".format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
file_path = os.path.join(log_path, file_name)
f= open(file_path,"w")
while True:
    recv = ser.read(ser.inWaiting())
    if not recv =="":
        s = recv.decode()
        lines = s.split()
        # print(lines)
        for i in lines:
            if i.startswith('$GNRMC'):
                print(i)
                f.write(i+"\n")
                f.flush()
                l = i.split(",")
                # print(len(l))
                if len(l)<8:
                    continue
                if l[7] =="":
                    v = 0.
                else:
                    v = float(l[7])*1.852
                print("速度：{}km".format(v))
        # break
    time.sleep(1)
f.close()
    # ~ line = ser.readline()
    # ~ # 这里如果从头取的话，就会出现b‘，所以要从第三个字符进行读取
    # ~ line = str(str(line)[2:])
    # ~ # print(line)
    # ~ if line.startswith('$GPRMC'):
        # ~ line = line.replace("\\r\\n'","")
        # ~ # print(line)
        # ~ # lines = str(line).split(',')  # 将line以“，”为分隔符
        # ~ # print(lines)
        # ~ # lines = json.dumps(lines)
        # ~ dataSocket.send(line.encode())
        # ~ # break

        # ~ # 等待接收服务端的消息
        # ~ recved = dataSocket.recv(BUFLEN)
        # ~ # 如果返回空bytes，表示对方关闭了连接
        # ~ if not recved:
            # ~ break
        # ~ # 打印读取的信息
        # ~ # print(recved.decode())
        # ~ print(recved.decode("utf-8","ignore"))
        # ~ print(time.strftime("%H-%M-%S",time.localtime()))
        
