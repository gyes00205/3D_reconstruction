import sys
import pyzed.sl as sl
import pandas as pd
import cv2
import numpy as np

if __name__ == "__main__":

    init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
                                 depth_mode=sl.DEPTH_MODE.PERFORMANCE)
    # init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode

    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    # if len(sys.argv) == 2:
    #     filepath = sys.argv[1]
    #     print("Using SVO file: {0}".format(filepath))
    #     init_params.set_from_svo_file(filepath)

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)

    runtime = sl.RuntimeParameters()
    camera_pose = sl.Pose()

    camera_info = zed.get_camera_information()

    py_translation = sl.Translation()
    pose_data = sl.Transform()

    image = sl.Mat()
    depth = sl.Mat()
    image_depth_zed = sl.Mat()
    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            tracking_state = zed.get_position(camera_pose)
            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                # Retrieve left image
                zed.retrieve_image(image, sl.VIEW.LEFT)
                # Retrieve depth map. Depth is aligned on the left image
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                # Retrieve the normalized depth image
                zed.retrieve_image(image_depth_zed, sl.VIEW.DEPTH)
                h, w = image.get_data().shape[:2]
                cv2.imshow('RGB', cv2.resize(image.get_data(), (w//2, h//2)))
                cv2.imshow('Depth', cv2.resize(image_depth_zed.get_data(), (w//2, h//2)))
                cv2.imshow('Depth Measure', cv2.resize(depth.get_data(), (w//2, h//2)))
                
                k = cv2.waitKey(1)
                pose_data = camera_pose.pose_data(sl.Transform())
                # print(pose_data)
                depth_np = (np.copy(depth.get_data())*1000).astype(np.uint16)
                print(depth.get_data().dtype)
                print(depth.get_data()[360,360])
                print(depth_np[360,360])
                cv2.imwrite('depth.png', depth_np)
                sys.stdout.flush()
                if k == ord('q'):
                    break
            
    cv2.destroyAllWindows()
    zed.close()