#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import rospy
import datetime
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import copy
import threading
from time import sleep

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
        self.data_guard = threading.Lock()
        
        # ros stuff
        self.bridge = CvBridge()
        
        #self.init_dai()
        #self.inited = False
        #self.h, self.w = None, None
        self.last_right_msg = None
        self.last_left_msg = None
        
        # pubs
        self.depth_pub = rospy.Publisher('~depth', Image, queue_size = 1)
        self.last_left_pub = rospy.Publisher('~last_left', Image, queue_size = 1)
        self.last_right_pub = rospy.Publisher('~last_right', Image, queue_size = 1)
        
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
        #stereo.syncedLeft.link(xoutLeft.input)
        #stereo.syncedRight.link(xoutRight.input)
        
        stereo.depth.link(xoutDepth.input)                
        #stereo.disparity.link(xoutDisparity.input)
        stereo.outConfig.link(xoutStereoCfg.input)
        
        stereo.setInputResolution(width, height)
        stereo.setRectification(False)                        
        
        rospy.loginfo('configured!')
    
    def stereo_cb(self, left_msg, right_msg):
        #if not self.inited:
            #self.init_dai(left_msg.width, left_msg.height)
        #if self.h is None or self.w is None:
            #self.h = left_msg.height
            #self.w = left_msg.width
            #return
        #if not self.inited:
            #return
        rospy.loginfo('got message')
        with self.data_guard:
            self.last_left_msg = left_msg
            self.last_right_msg = right_msg                                                
    
    def run(self):
        #rospy.sleep(10)
        while not rospy.is_shutdown():
            if not (self.last_left_msg is None and self.last_right_msg is None):
                self.init_dai(self.last_left_msg.width, self.last_right_msg.height)
                break
        #self.init_dai(640, 480)
                
        #rospy.sleep(5)
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
        
            #self.inited = True
            
            #currentConfig = None
            
            cnt = 0
            #timestamp_ms = 0
            #frame_interval_ms = 33
            while not rospy.is_shutdown():
                if self.last_left_msg is None and self.last_right_msg is None:
                    continue
                with self.data_guard:
                    msgs = [copy.copy(self.last_right_msg), copy.copy(self.last_left_msg)]
                    self.last_left_pub.publish(self.last_left_msg)
                    self.last_right_pub.publish(self.last_right_msg)
                    self.last_left_msg = None
                    self.last_right_msg = None
                
                
                for i, q in enumerate(self.in_q_list):
                    rospy.logwarn(f"sending {q.getName()}...")
                    data = self.bridge.imgmsg_to_cv2(msgs[i], desired_encoding="passthrough")
                    data = data.reshape(msgs[i].height*msgs[i].width)
                    
                    tstamp = datetime.timedelta(seconds = msgs[i].header.stamp.secs,
                                                milliseconds = msgs[i].header.stamp.nsecs * 1000000)
                    
                    #tstamp = datetime.timedelta(seconds = timestamp_ms // 1000,
                                                #milliseconds = timestamp_ms % 1000)
                    img = dai.ImgFrame()
                    img.setData(data)
                    img.setTimestamp(tstamp)
                    img.setInstanceNum(self.inStreamsCameraID[i])
                    img.setType(dai.ImgFrame.Type.RAW8)
                    img.setWidth(msgs[i].width)
                    img.setHeight(msgs[i].height)
                    q.send(img)
                    #if timestamp_ms == 0:  # Send twice for first iteration
                        #q.send(img)
                    #if self.first_it:
                        #q.send(img)
                        
                #timestamp_ms += frame_interval_ms
                #sleep(frame_interval_ms/1000)
                
                #rospy.sleep(0.1)
                rospy.logwarn("get configing...")
                #self.first_it = False
                #if currentConfig is None:
                currentConfig = self.inCfg.get()

                #lrCheckEnabled = currentConfig.get().algorithmControl.enableLeftRightCheck
                #extendedEnabled = currentConfig.get().algorithmControl.enableExtended
                #queues = q_list.copy()            
                rospy.logwarn("getting...")
                for q in self.q_list:            
                    data = q.get()
                    
                    frame = convertToCv2Frame(q.getName(), data, currentConfig, msgs[0].width)
                    depth_msg = self.bridge.cv2_to_imgmsg(frame, encoding='mono8')
                    self.depth_pub.publish(depth_msg)
                
                cnt+=1
                rospy.loginfo(f'{cnt} message proceeded')
                
                
                
        #rospy.spin()
        
if __name__ == '__main__':
    
    sen = StereoExtractorNode()
    sen.run()
        
