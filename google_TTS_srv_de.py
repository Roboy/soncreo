#!/usr/bin/python3

from google.cloud import texttospeech

from pydub import AudioSegment
from pydub.playback import play
import io

from roboy_cognition_msgs.srv import Talk, TalkResponse,  TalkToFile, TalkToFileResponse
from roboy_cognition_msgs.msg import SpeechSynthesis
import rospy
import rospkg
import rosgraph
# from rclpy.node import Node

import time
import logging

import sys
import os
sys.tracebacklimit = 0

if not rosgraph.is_master_online():
    raise Exception("ROS master is not online")

# export GOOGLE_APPLICATION_CREDENTIALS=""

class GoogleTTS():

    def __init__(self):
        rospy.init_node('google_tts_de')
        rospack = rospkg.RosPack()
        self.path = rospack.get_path('soncreo')+'/generated/'
        self.publisher = rospy.Publisher('/roboy/cognition/speech/synthesis', SpeechSynthesis)
        self.srv = rospy.Service('/roboy/cognition/speech/synthesis/talk/german', Talk,  self.talk_callback)
        self.save_srv = rospy.Service('/roboy/cognition/speech/synthesis/save/german', TalkToFile, self.save_callback)
        self.client = texttospeech.TextToSpeechClient()
        self.voice = texttospeech.types.VoiceSelectionParams(
            language_code='de-DE',
            name='de-DE-Wavenet-C')
            # ssml_gender=texttospeech.enums.SsmlVoiceGender.FEMALE)
        self.audio_config = texttospeech.types.AudioConfig(
            # effects_profile_id=["telephony-class-application"],
            pitch=3.2,
            speaking_rate=0.9,
            audio_encoding=texttospeech.enums.AudioEncoding.MP3)
        self.FORMAT = "wav"
        rospy.loginfo("Ready to /roboy/cognition/speech/synthesis/talk/german")
        self.synthesize("ich bin am start")

    def talk_callback(self, request):
        response = TalkResponse()
        response.success = True  # evtl.  return {'success':True}
        rospy.loginfo('Incoming Text: %s' % (request.text))
        msg = SpeechSynthesis()
        msg.duration = 5
        msg.phoneme = 'o'
        self.publisher.publish(msg)
        self.synthesize(request.text)
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
            response.success = True
            self.synthesize(request.text, request.filename)
        else:
            response.success = False
        return response

    @staticmethod
    def play_audio(fname):
        # from pydub.playback import play
        # from pydub import AudioSegment
        # song = AudioSegment.from_file(fname, format="wav")
        # import pdb; #pdb.set_trace()
        wf = wave.open(fname, 'rb')
        p = pyaudio.PyAudio()

        chunk = 1024

        # open stream based on the wave object which has been input.
        #import pdb; pdb.set_trace()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        input_device_index=1,
                        output=True)

        # read data (based on the chunk size)
        data = wf.readframes(chunk)

        # play stream (looping from beginning of file to the end)
        while len(data) > 0:
            # writing to the stream is what *actually* plays the sound.
            stream.write(data)
            data = wf.readframes(chunk)

            # cleanup stuff.
        stream.stop_stream()
        stream.close()
        p.terminate()

        print("Output wave generated")


    def synthesize(self, text, filename=None):
        synthesis_input = texttospeech.types.SynthesisInput(text=text)
        rospy.loginfo("sending request..")
        response = self.client.synthesize_speech(synthesis_input, self.voice, self.audio_config)
        # import pdb; pdb.set_trace()
        # print(len(response.audio_content))
        song = AudioSegment.from_file(io.BytesIO(response.audio_content), format="mp3")
        if filename:
                if "/" in filename:
                    dirname = filename.split("/")[0]
                    if not os.path.exists(self.path+dirname):
                        os.mkdir(self.path+dirname)
                song.export(self.path+filename+"."+self.FORMAT, format=self.FORMAT)
                rospy.loginfo("Saved to %s"%(self.path+filename+"."+self.FORMAT))
        play(song)
        # import pdb; pdb.set_trace()
        # song.export("google-output.wav", format="wav")
        # self.play_audio("google-output.wav")
        # song = AudioSegment.from_file("./output/google-output.wav", format="wav")
        # play(song)
        # self.get_logger().info("PyAudio now..")
        # Comb.play_audio("./output/google-output.wav")


def main(args=None):
    google_tts = GoogleTTS()

    rospy.spin()


if __name__ == '__main__':
    main()
