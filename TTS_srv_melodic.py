from roboy_cognition_msgs.srv import Talk, TalkResponse
from roboy_cognition_msgs.msg import SpeechSynthesis
from combine import Comb
import rospy


class Soncreo_TTS(Node):

    def __init__(self):
        rospy.init_node('soncreo_tts')
        self.publisher = rospy.Publisher('/roboy/cognition/speech/synthesis', SpeechSynthesis)
        self.srv = rospy.Service('/roboy/cognition/speech/synthesis/talk', Talk,  self.talk_callback)
        

        self.c=Comb()
        self.c.inference_audio("Speech synthesis is ready now")
        rospy.loginfo("Ready to /roboy/cognition/speech/synthesis/talk")

    def talk_callback(self, request):
        response = TalkResponse()
        response.success = True  # evtl.  return {'success':True}
        rospy.loginfo('Incoming Text: %s' % (request.text))
        msg = SpeechSynthesis()
        msg.duration = 5
        msg.phoneme = 'o'
        self.publisher.publish(msg)
        try:
            self.c.inference_audio(request.text)
        except:
            pass
        msg.phoneme = 'sil'
        msg.duration = 0
        self.publisher.publish(msg)
        return response


def main(args=None):

    soncreo_tts = Soncreo_TTS()
    
    while not rospy.is_shutdown:
        rospy.spin()


if __name__ == '__main__':
    main()
