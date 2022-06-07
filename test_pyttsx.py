# -*- coding: UTF-8 -*-
# sudo apt-get update && sudo apt-get install espeak && sudo pip install pyttsx3
# sudo apt update && sudo apt install espeak ffmpeg libespeak1
import pyttsx3
import sys

def use_pyttsx3():
    # 创建对象
    engine = pyttsx3.init()
    # 获取当前语音速率
    rate = engine.getProperty('rate')
    print(f'语音速率：{rate}')
    # 设置新的语音速率
    engine.setProperty('rate', 200)
    # 获取当前语音音量
    volume = engine.getProperty('volume')
    print(f'语音音量：{volume}')
    # 设置新的语音音量，音量最小为 0，最大为 1
    engine.setProperty('volume', 1)
    # 获取当前语音声音的详细信息
    voices = engine.getProperty('voices')
    print(f'语音声音详细信息：{voices}')
    print("共有",len(voices),"声音")
    # 设置当前语音声音为女性，当前声音不能读中文
    engine.setProperty('voice', voices[1].id)
    # 设置当前语音声音为男性，当前声音可以读中文
    # engine.setProperty('voice', voices[0].id)
    # 获取当前语音声音
    voice = engine.getProperty('voice')
    print(f'语音声音：{voice}')
    # 语音文本
    # ~ path = 'test.txt'
    # ~ with open(path, encoding='utf-8') as f_name:
        # ~ words = str(f_name.readlines()).replace(r'\n', '')
    # ~ # 将语音文本说出来
    # ~ for voice in voices:
        # ~ engine.setProperty('voice', voice.id)
        # ~ engine.say('The quick brown fox jumped over the lazy dog.')
        # ~ engine.runAndWait()
    words = "你好"
    engine.say(words)
    engine.runAndWait()
    words = "hello"
    engine.say(words)
    engine.runAndWait()
    engine.stop()

def test():
    # initialization

    engine = pyttsx3.init()
    # ~ voices = engine.getProperty('voices')
    # ~ print(f'语音声音详细信息：{voices}')
    # ~ engine.setProperty('voice', voices[1].id)
    # engine.setProperty('voice', "com.apple.speech.synthesis.voice.ting-ting.premium")
    # testing
    engine.setProperty('voice', 'zh')
    print(f'语音声音详细信息：{engine.getProperty("voice")}')
    engine.setProperty('rate', 300)
    engine.say('请保持注意力!')
    engine.runAndWait()
    # ~ engine.say("你好")
    # ~ engine.runAndWait()
    # ~ engine.say('请保持注意力!'.encode("utf-8"))
    # engine.say('this is my first text to speech converter program')
    # engine.say('everything i type here will be spoken by the program')

    

if __name__ == '__main__':
    # print(sys.getdefaultencoding())
    # ~ use_pyttsx3()
    test()
    
