#Start ROS2 service client for Soncreo ROS2
from roboy_cognition_msgs.srv import Talk
from combine import Comb as C
#g_node = None  # global Node
#



# Channel event callback function
class Soncreo_TTS():

    def __init__(self):
        # load models
        pass

    def talk_callback(request, response):

        response.success = True  # evtl.  return {'success':True} from Cerevoice
        g_node.get_logger().info('Incoming Text: %s' % (request.text))
        C().inference_audio(request.text)
        return response


def main(args=None):
     #instance with inference audio function

    global g_node
    try:
        import rclpy
        from roboy_cognition_msgs.srv import Talk
    except:
        print("Roboy_cognition_msgs was not found")

    # Init Rclpy
    rclpy.init(args=args)

    # create node
    g_node = rclpy.create_node('minimal_service')

    # create service
    srv = g_node.create_service(Talk, '/roboy/cognition/speech/synthesis/talk', Soncreo_TTS.talk_callback)
    print("Ready to /roboy/cognition/speech/synthesis/talk")

    # Speech Synthesis is ready now.
    C().inference_audio("Speech synthesis is ready now")

    # loop RCLpy
    while rclpy.ok():
        rclpy.spin_once(g_node)

    # Destroy Service
    g_node.destroy_service(srv)
    rclpy.shutdown()


if __name__ == '__main__':

    main()
