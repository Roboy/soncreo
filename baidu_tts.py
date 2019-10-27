#!/usr/bin/python3
# coding=utf-8
import sys
import json
import os
import time

from threading import Thread

from pydub import AudioSegment
from pydub.playback import play
import io

import rospy
import rospkg
import rosgraph
from roboy_cognition_msgs.srv import Talk, TalkResponse
from roboy_cognition_msgs.srv import TalkToFile, TalkToFileResponse
from roboy_cognition_msgs.msg import SpeechSynthesis

IS_PY3 = sys.version_info.major == 3
if IS_PY3:
    from urllib.request import urlopen
    from urllib.request import Request
    from urllib.error import URLError
    from urllib.parse import urlencode
    from urllib.parse import quote_plus
else:
    import urllib2
    from urllib import quote_plus
    from urllib2 import urlopen
    from urllib2 import Request
    from urllib2 import URLError
    from urllib import urlencode

sys.tracebacklimit = 0

if not rosgraph.is_master_online():
    raise Exception("ROS master is not online")

class DemoError(Exception):
    pass


class BaiduTTS():

    def __init__(self, api_key, secret_key):

        rospy.init_node('baidu_tts')
        rospack = rospkg.RosPack()
        self.path = rospack.get_path('soncreo')+'/generated/'
        self.publisher = rospy.Publisher('/roboy/cognition/speech/synthesis', SpeechSynthesis, queue_size=1)
        self.srv = rospy.Service('/roboy/cognition/speech/synthesis/talk/chinese', Talk,  self.talk_callback)
        self.en_srv = rospy.Service('/roboy/cognition/speech/synthesis/talk/english/baidu', Talk,  self.talk_callback)
        self.save_srv = rospy.Service('/roboy/cognition/speech/synthesis/save/chinese', TalkToFile, self.save_callback)
        self.API_KEY = api_key
        self.SECRET_KEY = secret_key

        # TEXT = "我期待着访问上海。 英飞凌是我最喜欢的公司"
        # 发音人选择, 基础音库：0为度小美，1为度小宇，3为度逍遥，4为度丫丫，
        # 精品音库：5为度小娇，103为度米朵，106为度博文，110为度小童，111为度小萌，默认为度小美
        self.PER = 103
        # 语速，取值0-15，默认为5中语速
        self.SPD = 5
        # 音调，取值0-15，默认为5中语调
        self.PIT = 1
        # 音量，取值0-9，默认为5中音量
        self.VOL = 5
        # 下载的文件格式, 3：mp3(default) 4： pcm-16k 5： pcm-8k 6. wav
        self.AUE = 6

        FORMATS = {3: "mp3", 4: "pcm", 5: "pcm", 6: "wav"}
        self.FORMAT = FORMATS[self.AUE]

        self.CUID = "123456PYTHON"

        self.TTS_URL = 'http://tsn.baidu.com/text2audio'

        self.TOKEN_URL = 'http://openapi.baidu.com/oauth/2.0/token'
        self.SCOPE = 'audio_tts_post'  # 有此scope表示有tts能力，没有请在网页里勾选

        self.token, self.token_expires_in = self.fetch_token()
        self.token_received = time.time()
        tt = Thread(target=self.token_watcher, args=())
        tt.setDaemon(True)
        tt.start()

        self.synthesize('你好人')

        rospy.loginfo("Baidu TTS is ready")


    def fetch_token(self):
        rospy.loginfo("fetch token begin")
        params = {'grant_type': 'client_credentials',
                  'client_id': self.API_KEY,
                  'client_secret': self.SECRET_KEY}
        post_data = urlencode(params)
        if (IS_PY3):
            post_data = post_data.encode('utf-8')
        req = Request(self.TOKEN_URL, post_data)
        try:
            f = urlopen(req, timeout=5)
            result_str = f.read()
        except URLError as err:
            print('token http response http code : ' + str(err.code))
            result_str = err.read()
        if (IS_PY3):
            result_str = result_str.decode()

        # print(result_str)
        result = json.loads(result_str)
        # print(result)
        if ('access_token' in result.keys() and 'scope' in result.keys()):
            if not self.SCOPE in result['scope'].split(' '):
                raise DemoError('scope is not correct')
            print('SUCCESS WITH TOKEN: %s ; EXPIRES IN SECONDS: %s' % (result['access_token'], result['expires_in']))
            return result['access_token'], result['expires_in']
        else:
            raise DemoError('MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not found in token response')


    def token_watcher(self):
        while True:
            if time.time() - self.token_received > self.token_expires_in - 5:
                rospy.loginfo("refreshing token")
                self.token, self.token_expires_in = self.fetch_token()
                self.token_received = time.time()


    def synthesize(self, text, filename=None, language='zh'):
        tex = quote_plus(text)  # 此处TEXT需要两次urlencode
        # print(tex)
        params = {'tok': self.token, 'tex': tex, 'per': self.PER, 'spd': self.SPD, 'pit': self.PIT, 'vol': self.VOL, 'aue': self.AUE, 'cuid': self.CUID,
                  'lan': language, 'ctp': 1}  # lan ctp 固定参数

        data = urlencode(params)
        # print('test on Web Browser' + TTS_URL + '?' + data)
        rospy.loginfo("fetching data")
        req = Request(self.TTS_URL, data.encode('utf-8'))
        has_error = False
        try:
            # print("urlopen")
            f = urlopen(req)
            result_str = f.read()

            headers = dict((name.lower(), value) for name, value in f.headers.items())

            has_error = ('content-type' not in headers.keys() or headers['content-type'].find('audio/') < 0)
        except  URLError as err:
            rospy.logerr('asr http response http code : ' + str(err.code))
            result_str = err.read()
            has_error = True

        if has_error:
            if (IS_PY3):
                result_str = str(result_str, 'utf-8')
            rospy.logerr("baidu tts api  error:" + result_str)
            return False
        else:
            song = AudioSegment.from_file(io.BytesIO(result_str), format=self.FORMAT)
            if filename:
                if "/" in filename:
                    dirname = filename.split("/")[0]
                    if not os.path.exists(self.path+dirname):
                        os.mkdir(self.path+dirname)
                song.export(self.path+filename+"."+self.FORMAT, format=self.FORMAT)
                rospy.loginfo("Saved to %s"%(self.path+filename+"."+self.FORMAT))
            play(song)
            rospy.loginfo("synthesize done")
            return True

    def talk_callback(self, request):
        response = TalkResponse()
        rospy.loginfo('Incoming Text: %s' % (request.text))
        msg = SpeechSynthesis()
        msg.duration = 5
        msg.phoneme = 'o'
        self.publisher.publish(msg)
        response.success = self.synthesize(request.text)
        msg.phoneme = 'sil'
        msg.duration = 0
        self.publisher.publish(msg)
        return response


    def save_callback(self, request):
        response = TalkToFileResponse()
        if request.text != "":
            rospy.loginfo('Incoming Text: %s' % (request.text))
            if request.filename != "":
                rospy.loginfo('Saving to filename: %s' % (request.filename))
            response.success = self.synthesize(request.text, request.filename)
        else:
            response.success = False
        return response

if __name__ == '__main__':
    API_KEY = os.environ.get('BAIDU_TTS_API_KEY')
    SECRET_KEY = os.environ.get('BAIDU_TTS_SECRET_KEY')

    if not API_KEY or not SECRET_KEY:
        raise ValueError('Could not get BAIDU_TTS_API_KEY or BAIDU_TTS_SECRET_KEY from environmnet')

    BaiduTTS(API_KEY, SECRET_KEY)
    rospy.spin()
    #
    # token = fetch_token()
    # tex = quote_plus(TEXT)  # 此处TEXT需要两次urlencode
    # print(tex)
    # params = {'tok': token, 'tex': tex, 'per': PER, 'spd': SPD, 'pit': PIT, 'vol': VOL, 'aue': AUE, 'cuid': CUID,
    #           'lan': 'zh', 'ctp': 1}  # lan ctp 固定参数
    #
    # data = urlencode(params)
    # print('test on Web Browser' + TTS_URL + '?' + data)
    #
    # req = Request(TTS_URL, data.encode('utf-8'))
    # has_error = False
    # try:
    #     print("urlopen")
    #     f = urlopen(req)
    #     result_str = f.read()
    #
    #     headers = dict((name.lower(), value) for name, value in f.headers.items())
    #
    #     has_error = ('content-type' not in headers.keys() or headers['content-type'].find('audio/') < 0)
    # except  URLError as err:
    #     print('asr http response http code : ' + str(err.code))
    #     result_str = err.read()
    #     has_error = True
    #
    # save_file = "error.txt" if has_error else 'result.' + FORMAT
    # with open(save_file, 'wb') as of:
    #     of.write(result_str)
    #
    # if has_error:
    #     if (IS_PY3):
    #         result_str = str(result_str, 'utf-8')
    #     print("tts api  error:" + result_str)
    #
    # print("result saved as :" + save_file)
