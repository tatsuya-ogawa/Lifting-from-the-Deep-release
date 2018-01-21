#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'
"""

import __init__

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose

import cv2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from os.path import dirname, realpath

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
IMAGE_FILE_PATH = PROJECT_PATH + '/data/images/test_image.png'
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'


def main():

    fig = plt.figure()
    ax2d = fig.add_subplot(1,1,1)
    ax2d.axis('off')
    fig = plt.figure()
    ax3d = fig.add_subplot(1,1,1, projection='3d')

    cap = cv2.VideoCapture('ohayo.mp4')
    pose_estimator = None
    while(True):
        # image = cv2.imread(IMAGE_FILE_PATH)
        for _ in range(60):
            ret,image = cap.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb

        # create pose estimator
        image_size = image.shape
        if pose_estimator is None:
            pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)

            # load model
            pose_estimator.initialise()

        # estimation
        pose_2d, visibility, pose_3d = pose_estimator.estimate(image)

        # # close model
        # pose_estimator.close()

        # Show 2D and 3D poses
        display_results(image, pose_2d, visibility, pose_3d,ax2d,ax3d)

        #plt.show()
        plt.pause(.01)

def display_results(in_image, data_2d, joint_visibility, data_3d, ax2d, ax3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    draw_limbs(in_image, data_2d, joint_visibility)
    ax2d.imshow(in_image)
    # ax2d.axis('off')

    if data_3d is not None:
        # # Show 3D poses
        # for single_3D in data_3d:
        #     # or plot_pose(Prob3dPose.centre_all(single_3D))
        #     plot_pose(single_3D, ax3d)
        plot_pose(data_3d[0],ax3d)

if __name__ == '__main__':
    import sys
    sys.exit(main())
