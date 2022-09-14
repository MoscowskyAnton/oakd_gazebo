#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import rospy
import datetime
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def convertToCv2Frame(name, image, config, width):
    
    baseline = 75
    fov = 71.86
    focal = width / (2 * np.tan(fov / 2 / 180 * np.pi))

    maxDisp = config.getMaxDisparity()
    subpixelLevels = pow(2, config.get().algorithmControl.subpixelFractionalBits)
    subpixel = config.get().algorithmControl.enableSubpixel
    dispIntegerLevels = maxDisp if not subpixel else maxDisp / subpixelLevels

    frame = image.getFrame()

    # frame.tofile(name+".raw")

    if name == 'depth':
        dispScaleFactor = baseline * focal
        with np.errstate(divide='ignore'):
            frame = dispScaleFactor / frame

        frame = (frame * 255. / dispIntegerLevels).astype(np.uint8)
        #frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
    elif 'confidence_map' in name:
        pass
    elif name == 'disparity_cost_dump':
        # frame.tofile(name+'.raw')
        pass
    elif 'disparity' in name:
        if 1: # Optionally, extend disparity range to better visualize it
            frame = (frame * 255. / maxDisp).astype(np.uint8)
        # if 1: # Optionally, apply a color map
        #     frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)

    return frame

class StereoExtractorNode(object):
    
    def __init__(self):
        
        rospy.init_node('stereo_extractor_node')
        
        # params
        self.lrcheck =  rospy.get_param('~lrcheck', True)
        self.extended = rospy.get_param('~extended', False)
        self.subpixel = rospy.get_param('~subpixel', True)
        
        # vars
        self.first_it = True
        
        # ros stuff
        self.bridge = CvBridge()
        
        #self.init_dai()
        self.inited = False
        
        # pubs
        self.depth_pub = rospy.Publisher('~depth', Image, queue_size = 1)
        
        # subs
        left_sub = message_filters.Subscriber('left', Image)
        right_sub = message_filters.Subscriber('right', Image)
        
        ts = message_filters.ApproximateTimeSynchronizer([left_sub, right_sub], 10, 0.1, allow_headerless=False)
        
        #rospy.Subscriber('left/camera_info', )
        
        rospy.loginfo('waiting for messages...')
        ts.registerCallback(self.stereo_cb)
        
    def init_dai(self, width, height):
        rospy.loginfo('configuring...')
        self.pipeline = dai.Pipeline()
    
        stereo = self.pipeline.create(dai.node.StereoDepth)
        
        monoLeft = self.pipeline.create(dai.node.XLinkIn)
        monoRight = self.pipeline.create(dai.node.XLinkIn)
        xinStereoDepthConfig = self.pipeline.create(dai.node.XLinkIn)
        
        xoutLeft = self.pipeline.create(dai.node.XLinkOut)
        xoutRight = self.pipeline.create(dai.node.XLinkOut)
        xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        xoutConfMap = self.pipeline.create(dai.node.XLinkOut)
        xoutDisparity = self.pipeline.create(dai.node.XLinkOut)
        xoutRectifLeft = self.pipeline.create(dai.node.XLinkOut)
        xoutRectifRight = self.pipeline.create(dai.node.XLinkOut)
        xoutStereoCfg = self.pipeline.create(dai.node.XLinkOut)
        
        xinStereoDepthConfig.setStreamName("stereoDepthConfig")
        monoLeft.setStreamName('in_left')
        monoRight.setStreamName('in_right')

        xoutLeft.setStreamName('left')
        xoutRight.setStreamName('right')
        xoutDepth.setStreamName('depth')
        xoutConfMap.setStreamName('confidence_map')
        xoutDisparity.setStreamName('disparity')
        xoutRectifLeft.setStreamName('rectified_left')
        xoutRectifRight.setStreamName('rectified_right')
        xoutStereoCfg.setStreamName('stereo_cfg')
        
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
        stereo.setLeftRightCheck(self.lrcheck)
        stereo.setExtendedDisparity(self.extended)
        stereo.setSubpixel(self.subpixel)
        
        stereo.setRuntimeModeSwitch(True)
        
        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        xinStereoDepthConfig.out.link(stereo.inputConfig)
        stereo.syncedLeft.link(xoutLeft.input)
        stereo.syncedRight.link(xoutRight.input)
        
        stereo.depth.link(xoutDepth.input)                
        stereo.disparity.link(xoutDisparity.input)
        stereo.outConfig.link(xoutStereoCfg.input)
        
        stereo.setInputResolution(width, height)
        stereo.setRectification(False)
                
        streams = ['depth']
        
        self.device = dai.Device(self.pipeline)
        
        stereoDepthConfigInQueue = self.device.getInputQueue("stereoDepthConfig")
        inStreams = ['in_right', 'in_left']
        self.inStreamsCameraID = [dai.CameraBoardSocket.RIGHT, dai.CameraBoardSocket.LEFT]
        
        self.in_q_list = []
        for s in inStreams:
            q = self.device.getInputQueue(s)
            self.in_q_list.append(q)
        
        # Create a receive queue for each stream
        self.q_list = []
        for s in streams:
            q = self.device.getOutputQueue(s, 8, blocking=False)
            self.q_list.append(q)
            
        self.inCfg = self.device.getOutputQueue("stereo_cfg", 8, blocking=False)
        
        self.inited = True
        rospy.loginfo('configured!')
    
    def stereo_cb(self, left_msg, right_msg):
        if not self.inited:
            self.init_dai(left_msg.width, left_msg.height)
                    
        rospy.loginfo('got message')
        msgs = [right_msg, left_msg]
        for i, q in enumerate(self.in_q_list):
            data = self.bridge.imgmsg_to_cv2(msgs[i], desired_encoding="passthrough")
            data = data.reshape(msgs[i].height*msgs[i].width)
            
            tstamp = datetime.timedelta(seconds = msgs[i].header.stamp.secs,
                                        milliseconds = msgs[i].header.stamp.nsecs * 1000000)
            
            img = dai.ImgFrame()
            img.setData(data)
            img.setTimestamp(tstamp)
            img.setInstanceNum(self.inStreamsCameraID[i])
            img.setType(dai.ImgFrame.Type.RAW8)
            img.setWidth(msgs[i].width)
            img.setHeight(msgs[i].height)
            q.send(img)
            if self.first_it:
                q.send(img)
        
        self.first_it = False
        currentConfig = self.inCfg.get()

        lrCheckEnabled = currentConfig.get().algorithmControl.enableLeftRightCheck
        extendedEnabled = currentConfig.get().algorithmControl.enableExtended
        #queues = q_list.copy()            
        
        for q in self.q_list:            
            data = q.get()
            frame = convertToCv2Frame(q.getName(), data, currentConfig, left_msg.width)
            depth_msg = self.bridge.cv2_to_imgmsg(frame, encoding='mono8')
            self.depth_pub.publish(depth_msg)
            
        rospy.loginfo('message proceeded')
            
    
    def run(self):
        rospy.spin()
        
if __name__ == '__main__':
    
    sen = StereoExtractorNode()
    sen.run()
        
