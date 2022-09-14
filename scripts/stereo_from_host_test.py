#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import argparse
import datetime
from time import sleep
import matplotlib.pyplot as plt

width = 640
height = 480
baseline = 75
fov = 71.86
focal = width / (2 * np.tan(fov / 2 / 180 * np.pi))

def convertToCv2Frame(name, image, config):

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
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', nargs='?', help="Path to recorded frames")
    args = parser.parse_args()
    
    # StereoDepth initial config options.
    outDepth = True  # Disparity by default
    outConfidenceMap = False  # Output disparity confidence map
    outRectified = False   # Output and display rectified streams
    lrcheck = True   # Better handling for occlusions
    extended = False  # Closer-in minimum depth, disparity range is doubled. Unsupported for now.
    subpixel = True   # Better accuracy for longer distance, fractional disparity 32-levels
    
    
    
    # Pipeline
    
    pipeline = dai.Pipeline()
    
    stereo = pipeline.create(dai.node.StereoDepth)
    
    monoLeft = pipeline.create(dai.node.XLinkIn)
    monoRight = pipeline.create(dai.node.XLinkIn)
    xinStereoDepthConfig = pipeline.create(dai.node.XLinkIn)
    
    xoutLeft = pipeline.create(dai.node.XLinkOut)
    xoutRight = pipeline.create(dai.node.XLinkOut)
    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutConfMap = pipeline.create(dai.node.XLinkOut)
    xoutDisparity = pipeline.create(dai.node.XLinkOut)
    xoutRectifLeft = pipeline.create(dai.node.XLinkOut)
    xoutRectifRight = pipeline.create(dai.node.XLinkOut)
    xoutStereoCfg = pipeline.create(dai.node.XLinkOut)
    
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
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)
    
    stereo.setRuntimeModeSwitch(True)
    
    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    xinStereoDepthConfig.out.link(stereo.inputConfig)
    stereo.syncedLeft.link(xoutLeft.input)
    stereo.syncedRight.link(xoutRight.input)
    if outDepth:
        stereo.depth.link(xoutDepth.input)
    if outConfidenceMap:
        stereo.confidenceMap.link(xoutConfMap.input)
    stereo.disparity.link(xoutDisparity.input)
    if outRectified:
        stereo.rectifiedLeft.link(xoutRectifLeft.input)
        stereo.rectifiedRight.link(xoutRectifRight.input)
    stereo.outConfig.link(xoutStereoCfg.input)
    
    stereo.setInputResolution(width, height)
    stereo.setRectification(False)
    
    streams = ['left', 'right']
    if outRectified:
        streams.extend(['rectified_left', 'rectified_right'])
    streams.append('disparity')
    if outDepth:
        streams.append('depth')
    if outConfidenceMap:
        streams.append('confidence_map')    
    
    with dai.Device(pipeline) as device:
        
        stereoDepthConfigInQueue = device.getInputQueue("stereoDepthConfig")
        inStreams = ['in_right', 'in_left']
        inStreamsCameraID = [dai.CameraBoardSocket.RIGHT, dai.CameraBoardSocket.LEFT]
        in_q_list = []
        for s in inStreams:
            q = device.getInputQueue(s)
            in_q_list.append(q)

        # Create a receive queue for each stream
        q_list = []
        for s in streams:
            q = device.getOutputQueue(s, 8, blocking=False)
            q_list.append(q)


        inCfg = device.getOutputQueue("stereo_cfg", 8, blocking=False)
                        

        # Need to set a timestamp for input frames, for the sync stage in Stereo node
        timestamp_ms = 0
        episode = 2
        index = 0
        prevQueues = q_list.copy()
        while True:
            # Handle input streams, if any
            if in_q_list:
                dataset_size = 10  # Number of image pairs
                frame_interval_ms = 50
                for i, q in enumerate(in_q_list):
                    path = args.dataset + f'/{episode}_' + str(index) + '_' + q.getName() + '.png'
                    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    data = cv2.resize(data, (width, height), interpolation = cv2.INTER_AREA)
                    data = data.reshape(height*width)
                    tstamp = datetime.timedelta(seconds = timestamp_ms // 1000,
                                                milliseconds = timestamp_ms % 1000)
                    img = dai.ImgFrame()
                    img.setData(data)
                    img.setTimestamp(tstamp)
                    img.setInstanceNum(inStreamsCameraID[i])
                    img.setType(dai.ImgFrame.Type.RAW8)
                    img.setWidth(width)
                    img.setHeight(height)
                    q.send(img)
                    if timestamp_ms == 0:  # Send twice for first iteration
                        q.send(img)
                    # print("Sent frame: {:25s}".format(path), 'timestamp_ms:', timestamp_ms)
                timestamp_ms += frame_interval_ms
                index = (index + 1) % dataset_size
                sleep(frame_interval_ms / 1000)

            # Handle output streams
            currentConfig = inCfg.get()

            lrCheckEnabled = currentConfig.get().algorithmControl.enableLeftRightCheck
            extendedEnabled = currentConfig.get().algorithmControl.enableExtended
            queues = q_list.copy()            
            
            def ListDiff(li1, li2):
                return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

            diff = ListDiff(prevQueues, queues)
            for s in diff:
                name = s.getName()
                cv2.destroyWindow(name)
            prevQueues = queues.copy()

            for q in queues:
                if q.getName() in ['left', 'right']: continue
                data = q.get()
                frame = convertToCv2Frame(q.getName(), data, currentConfig)
                cv2.imshow(q.getName(), frame)
                if q.getName() == 'depth':
                    plt.imsave(args.dataset+f'/{episode}_{index}_oak_depth.png', frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
