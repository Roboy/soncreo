from roboy_cognition_msgs.srv import Talk
from roboy_cognition_msgs.msg import SpeechSynthesis
from combine import Comb
import rclpy
from rclpy.node import Node


class Soncreo_TTS(Node):

    def __init__(self):
        super().__init__('soncreo_tts')
        self.publisher = self.create_publisher(SpeechSynthesis, '/roboy/cognition/speech/synthesis')
        self.srv = self.create_service(Talk, '/roboy/cognition/speech/synthesis/talk', self.talk_callback)
        print("Ready to /roboy/cognition/speech/synthesis/talk")

        self.c=Comb()
        print("speech synthesis is ready now")
        self.c.inference_audio("Speech synthesis is ready now")

    def talk_callback(self, request, response):
        response.success = True  # evtl.  return {'success':True}
        self.get_logger().info('Incoming Text: %s' % (request.text))
        msg = SpeechSynthesis()
        msg.duration = 5
        msg.phoneme = 'o'
        self.publisher.publish(msg)
        self.c.inference_audio(request.text)
        msg.phoneme = 'sil'
        msg.duration = 0
        self.publisher.publish(msg)
        return response


def main(args=None):
    rclpy.init(args=args)

    soncreo_tts = Soncreo_TTS()

    while rclpy.ok():
        rclpy.spin_once(soncreo_tts)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
