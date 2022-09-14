#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import rospy
import datetime
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import copy
import threading
from time import sleep

def convertToCv2Frame(name, image, config, focal):
        
    frame = image.getFrame()    
    if name == 'depth':        
        frame = (frame).astype(np.uint16)
    return frame

class StereoExtractorNode(object):
    
    def __init__(self):
        
        rospy.init_node('stereo_extractor_node')
        
        # params
        self.lrcheck =  rospy.get_param('~lrcheck', True)
        self.extended = rospy.get_param('~extended', False)
        self.subpixel = rospy.get_param('~subpixel', True)
        #self.depth_frame = rospy.get_param('~depth_frame', "")
        
        # vars
        self.first_it = True
        self.data_guard = threading.Lock()
        
        # ros stuff
        self.bridge = CvBridge()
                
        self.last_right_msg = None
        self.last_left_msg = None
        self.last_info_msg = None
        
        # pubs
        self.depth_pub = rospy.Publisher('~depth', Image, queue_size = 1)        
        self.info_pub = rospy.Publisher('~camera_info', CameraInfo, queue_size = 1)        
        
        # subs
        left_sub = message_filters.Subscriber('left', Image)
        right_sub = message_filters.Subscriber('right', Image)
        info_sub = message_filters.Subscriber('right/info', CameraInfo)
        
        ts = message_filters.ApproximateTimeSynchronizer([left_sub, right_sub, info_sub], 10, 0.1, allow_headerless=False)                
        
        rospy.loginfo('waiting for messages...')
        ts.registerCallback(self.stereo_cb)
        
    def init_dai(self, width, height):
        rospy.loginfo('configuring...')
        self.pipeline = dai.Pipeline()
    
        stereo = self.pipeline.create(dai.node.StereoDepth)
        
        monoLeft = self.pipeline.create(dai.node.XLinkIn)
        monoRight = self.pipeline.create(dai.node.XLinkIn)
        xinStereoDepthConfig = self.pipeline.create(dai.node.XLinkIn)
                
        xoutDepth = self.pipeline.create(dai.node.XLinkOut)        
        xoutStereoCfg = self.pipeline.create(dai.node.XLinkOut)
        
        xinStereoDepthConfig.setStreamName("stereoDepthConfig")
        monoLeft.setStreamName('in_left')
        monoRight.setStreamName('in_right')
        
        xoutDepth.setStreamName('depth')        
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
        
        stereo.depth.link(xoutDepth.input)                        
        stereo.outConfig.link(xoutStereoCfg.input)
        
        stereo.setInputResolution(width, height)
        stereo.setRectification(False)                        
        
        rospy.loginfo('configured!')
    
    def stereo_cb(self, left_msg, right_msg, info_msg):                
        with self.data_guard:
            self.last_left_msg = left_msg
            self.last_right_msg = right_msg      
            self.last_info = info_msg
    
    def run(self):   
        # wait data for init
        while not rospy.is_shutdown():
            if not (self.last_left_msg is None and self.last_right_msg is None):
                self.init_dai(self.last_left_msg.width, self.last_right_msg.height)
                break        
                
        # processing
        with dai.Device(self.pipeline) as device:
            streams = ['depth']
        
            stereoDepthConfigInQueue = device.getInputQueue("stereoDepthConfig")
            inStreams = ['in_right', 'in_left']
            self.inStreamsCameraID = [dai.CameraBoardSocket.RIGHT, dai.CameraBoardSocket.LEFT]
        
            self.in_q_list = []
            for s in inStreams:
                q = device.getInputQueue(s)
                self.in_q_list.append(q)
        
            # Create a receive queue for each stream
            self.q_list = []
            for s in streams:
                q = device.getOutputQueue(s, 30, blocking=False)
                self.q_list.append(q)
            
            self.inCfg = device.getOutputQueue("stereo_cfg", 30, blocking=False)
                                            
            calibData = device.readCalibration()
            M, _, _ = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.RIGHT)
            focal = M[0][0]
                                            
            while not rospy.is_shutdown():
                if self.last_left_msg is None and self.last_right_msg is None:
                    continue
                with self.data_guard:
                    msgs = [copy.copy(self.last_right_msg), copy.copy(self.last_left_msg)]                    
                    info = self.last_info
                    self.last_left_msg = None
                    self.last_right_msg = None                
                
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
                    
                
                currentConfig = self.inCfg.get()                                
                
                for q in self.q_list:         
                    if q.getName() == 'depth':
                        data = q.get().getFrame()                  
                        #frame = convertToCv2Frame(q.getName(), data, currentConfig, focal)
                        frame = data.astype(np.uint16)
                        depth_msg = self.bridge.cv2_to_imgmsg(frame, encoding='mono16')
                        depth_msg.header = info.header                    
                        
                        self.depth_pub.publish(depth_msg)
                        self.info_pub.publish(info)
                    
                                
        
if __name__ == '__main__':
    
    sen = StereoExtractorNode()
    sen.run()
        
