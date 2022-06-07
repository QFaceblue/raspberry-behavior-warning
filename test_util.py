import cv2
import numpy as np
import time
import os 
# source /opt/intel/openvino/bin/setupvars.sh
def detect_vpu():
	xml_path = r"weights/mobilenetv2_1_my_224.xml"
	bin_path = r"weights/mobilenetv2_1_my_224.bin"
	try:
		net = cv2.dnn.readNet(xml_path,bin_path)
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)    
		image = np.zeros([224,224,3],dtype=np.float32)
		blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (0, 0, 0), False)
		net.setInput(blob)
		probs = net.forward()
	except Exception as e:
		print(type(e))
		# if str(e).find("openvino")>0:
		if "openvino" in str(e):
			print("No vpu was detected!")
		else:
			print("OpenVINO environment is not initialized!")
		return False
	return True

def detect_vpu2():
	for index,line in enumerate(os.popen("lsusb")):
		if "Intel Movidius MyriadX" in line:
			print("Found vpu!")	
			return True
	print("No vpu was detected!")	
	return False

def detect_beidou():
	for index,line in enumerate(os.popen("lsusb")):
		if "QinHeng Electronics HL-340 USB-Serial adapter" in line:
			print("Found beidou!")	
			return True
	print("No beidou was detected!")	
	return False	
	
def get_mac():
	mac = []
	for index,line in enumerate(os.popen("ifconfig")):
		if "ether" in line:
			mac.append(line[14:31])
			# ~ print(line)
			# ~ print(line.find("ether"))
			# ~ print(line[14:31])
			# ~ print(index)
		
	return mac
	# ~ lines = os.popen("ifconfig").readlines()
	# ~ print(len(lines))

def get_temp():
	# ~ for index,line in enumerate(os.popen("/opt/vc/bin/vcgencmd measure_temp")):
		# ~ print(line)
	# ~ for index,line in enumerate(os.popen("cat /sys/class/thermal/thermal_zone0/temp")):
		# ~ print(line)	
	line = os.popen("cat /sys/class/thermal/thermal_zone0/temp").readline()
	return float(line)/1000.

def test_net():
	code = os.system('ping www.baidu.com -c 1')
	return code
	
if __name__ == '__main__':
	start = time.time()
	# print(detect_vpu())
	# print(detect_vpu2())
	print(detect_beidou())
	# print(test_net())
	# print(get_mac())
	# print(get_temp())
	print("total time:{}".format((time.time()-start)))
    
