#!/usr/bin/python3

from google.cloud import texttospeech

from pydub import AudioSegment
from pydub.playback import play
import io

from roboy_cognition_msgs.srv import Talk
from roboy_cognition_msgs.msg import SpeechSynthesis
import rclpy
from rclpy.node import Node

# export GOOGLE_APPLICATION_CREDENTIALS=""

class GoogleTTS(Node):

    def __init__(self):
        super().__init__('google_tts_de')
        self.publisher = self.create_publisher(SpeechSynthesis, '/roboy/cognition/speech/synthesis')
        self.srv = self.create_service(Talk, '/roboy/cognition/speech/synthesis/talk/german', self.talk_callback)

        self.client = texttospeech.TextToSpeechClient()
        self.voice = texttospeech.types.VoiceSelectionParams(
            language_code='de-DE',
            ssml_gender=texttospeech.enums.SsmlVoiceGender.FEMALE)
        self.audio_config = texttospeech.types.AudioConfig(
            # effects_profile_id=["large-home-entertainment-class-device"],
            # pitch=-1.0,
            audio_encoding=texttospeech.enums.AudioEncoding.MP3)

        self.get_logger().info("Ready to /roboy/cognition/speech/synthesis/talk/german")

    def talk_callback(self, request, response):
        response.success = True  # evtl.  return {'success':True}
        self.get_logger().info('Incoming Text: %s' % (request.text))
        msg = SpeechSynthesis()
        msg.duration = 5
        msg.phoneme = 'o'
        self.publisher.publish(msg)
        self.synthesize(request.text)
        msg.phoneme = 'sil'
        msg.duration = 0
        self.publisher.publish(msg)
        return response

    def synthesize(self, text):
        synthesis_input = texttospeech.types.SynthesisInput(text=text)

        response = self.client.synthesize_speech(synthesis_input, self.voice, self.audio_config)
        song = AudioSegment.from_file(io.BytesIO(response.audio_content), format="mp3")
        play(song)
        # song.export("13-kommher.wav", format="wav")


def main(args=None):
    rclpy.init(args=args)

    google_tts = GoogleTTS()

    while rclpy.ok():
        rclpy.spin_once(google_tts)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
