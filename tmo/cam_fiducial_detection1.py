from tmo.cam_to_file import h5_img_collect
from tmo.db import camviewer
import pcdsdevices.pim
from time import sleep, strftime
from os import path, mkdir, listdir
import matplotlib.pyplot as plt
import logging
import cv2 as cv
import numpy as np
import h5py

logger = logging.getLogger(__name__)
IMG_PATH = '/cds/home/opr/tmoopr/fiducial_detection'

def collect_fiducial_image(camera):
    camviewer_camera = camviewer[camera.name]
    try:
       if not camviewer_camera.cam.acquire_time.connected:
           logger.error('Cannot set camera acquire time!')
           return
    except Exception as e:
       logger.error(e)
       return
    target = ''
    if type(camera) is pcdsdevices.pim.PPM:
        camera.led.put(50.0)
        camera.detector.exposure.put(0.0004)
        target = 'YAG1'
    elif type(camera) is pcdsdevices.pim.XPIM or type(camera) is pcdsdevices.pim.IM2K0:
        camera.led.power.put(1)
        camera.detector.exposure.put(0.001)
        target = 'RETICLE'
    else:
        logger.error('Unsupported imager!')
        return
    camera.detector.acquire.put(1)

    try:
        if not camviewer_camera.cam.array_rate.connected:
            logger.error('Cannot get camera array rate!')
            if type(camera) is pcdsdevices.pim.PPM:
                camera.led.put(0)
            else:
                camera.led.power.put(0)
            camera.detector.acquire.put(0)
            return
    except Exception as e:
        logger.error(e)
        if type(camera) is pcdsdevices.pim.PPM:
            camera.led.put(0)
        else:
            camera.led.power.put(0)
        camera.detector.acquire.put(0)
        return

    counter = 0
    while camviewer_camera.cam.array_rate.get() < 0.5:
        if counter > 100:
            logger.error('Camera failed to reach 0.5 Hz within 10 seconds')
            if type(camera) is pcdsdevices.pim.PPM:
                camera.led.put(0)
            else:
                camera.led.power.put(0)
            camera.detector.acquire.put(0)
            return
        sleep(0.1)
        counter += 1

    if camera.target() == target:
        logger.info(f'Target is already at {target}')
    else:
        try:
            camera.target.mv(target, wait=True, timeout=20.0)
        except:
            logger.error(f'Failed to move to {target} within 20 seconds')
            if type(camera) is pcdsdevices.pim.PPM:
                camera.led.put(0)
            else:
                camera.led.power.put(0)
            camera.detector.acquire.put(0)
            return

    logger.info('Collecting image')
    sleep(2)

    save_path = path.join(IMG_PATH, camera.name) + '/'
    image_name = strftime('%Y-%m-%d-%H:%M:%S')
    h5_img_collect(camera.name, images=2, path=save_path, file_name=image_name)

    camera.target('OUT')
    if type(camera) is pcdsdevices.pim.PPM:
        camera.led.put(0)
    else:
        camera.led.power.put(0)
    camera.detector.acquire.put(0)

def analyze_image(image_path, show_image = True):
    f = h5py.File(image_path)
    #image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    image = cv.convertScaleAbs(f['entry/data/data'][0], alpha=1/16)
    # norm: normalize brightness of image
    norm = np.zeros(image.shape)
    norm = cv.normalize(image, norm, 0, 190, cv.NORM_MINMAX)
    # blur: reduce noise
    blur = cv.medianBlur(norm, 5)
    # adaptive threshold: changes image into binary bitmap
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,71,-3)
    # morph_open: more noise reduction
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, open_kernel)
    # morph_close: expand fiducials into full squares
    dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (12,12))
    dilate = cv.morphologyEx(opened, cv.MORPH_DILATE, dilate_kernel)
    # blob detector: determines coordinates of fiducials
    params = cv.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5000
    params.maxArea = 15000
    params.filterByColor = True
    params.blobColor = 255
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(dilate)
    #for keypoint in keypoints:
    #    logger.info(f'Keypoint size: {keypoint.size}')

    if show_image:
        f, axs = plt.subplots(1,2)
        annotated_image = cv.drawKeypoints(norm, keypoints, None, (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        axs[0].imshow(annotated_image)
        axs[1].imshow(dilate, cmap='gray')

    return cv.KeyPoint_convert(keypoints)

def analyze_image_orb(image_path, show_image=False):
    f = h5py.File(image_path)
    #image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    image = cv.convertScaleAbs(f['entry/data/data'][0], alpha=1/16)
    # eq: equalize image histogram
    eq = cv.equalizeHist(image)
    orb = cv.ORB_create()
    kp, des = orb.detectAndCompute(eq, None)
    kp_filtered = []
    des_filtered = []
    for index, keypoint in enumerate(kp):
        if keypoint.response >= 0.0005:
            kp_filtered.append(keypoint)
            des_filtered.append(des[index])
    if show_image:
        annotated_image = cv.drawKeypoints(eq, kp_filtered, None, (0,0,255), 0)
        plt.imshow(annotated_image)
    return kp_filtered, np.array(des_filtered), eq

def check_fiducial_change_orb(camera_name, show_image=True):
    image_dir = path.join(IMG_PATH, camera_name)
    images = sorted(listdir(path.join(image_dir)), reverse=True)
    if len(images) < 2:
        logger.error(f'Less than 2 images found in {path.join(IMG_PATH, camera_name)}')
        return
    kp1, des1, img1 = analyze_image_orb(path.join(image_dir, images[0]))
    kp2, des2, img2 = analyze_image_orb(path.join(image_dir, images[1]))

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key = lambda x:x.distance, reverse=True)
    diffs = []
    good_matches = []
    kp1_arr = cv.KeyPoint_convert(kp1)
    kp2_arr = cv.KeyPoint_convert(kp2)
    for match in matches:
        pt1 = kp1_arr[match.queryIdx]
        pt2 = kp2_arr[match.trainIdx]
        if match.distance <= 35:
            good_matches.append(match)
            diffs.append(pt1 - pt2)
    diffs_norm = np.linalg.norm(diffs, axis=1)
    if show_image:
        f, axs = plt.subplots(2,1)
        img3 = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        axs[0].imshow(img3)
        axs[1].hist(diffs_norm, bins=[0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000])
        axs[1].set_xscale('log')
    return np.sort(diffs_norm)

def microns_per_pixel(hull):
    sides = hull - np.roll(hull, 1, axis=0)
    average_length = np.average(np.linalg.norm(sides, axis=1))
    return 8500 / average_length

def check_fiducial_change(camera_name):
    image_dir = path.join(IMG_PATH, camera_name)
    images = sorted(listdir(path.join(image_dir)), reverse=True)
    if len(images) < 2:
        logger.error(f'Less than 2 images found in {path.join(IMG_PATH, camera_name)}')
        return

    new_keypoints = analyze_image(path.join(image_dir, images[0]), False)
    old_keypoints = analyze_image(path.join(image_dir, images[1]), False)
    if len(new_keypoints) != len(old_keypoints):
        logger.error(f'Number of fiducials detected does not match: {len(new_keypoints)} in new image vs {len(old_keypoints)} in old image')
        return

    num_keypoints = len(new_keypoints)
    # convex hull ensures keypoints are in counterclockwise order
    new_hull = np.squeeze(cv.convexHull(new_keypoints))
    old_hull = np.squeeze(cv.convexHull(old_keypoints))
    mpp = microns_per_pixel(new_hull)
    logger.info(f'Calculated microns per pixel: {mpp}')

    # align arrays so that the keypoints in matching indices are the closest to each other
    offset = 0
    min_distance = np.linalg.norm(new_hull[0] - old_hull[0])
    for i in range(1, num_keypoints):
        distance = np.linalg.norm(new_hull[0] - old_hull[i])
        if distance < min_distance:
            min_distance = distance
            offset = i
    new_hull = np.roll(new_hull, offset, axis=0)

    diffs = (new_hull - old_hull) * mpp
    norm_average = np.linalg.norm(np.average(diffs, axis=0))
    average_norm = np.average(np.linalg.norm(diffs, axis=1))
    logger.info(f'The average movement of all fiducials is {norm_average:.2f} microns')
    logger.info(f'Each fiducial has moved an average of {average_norm:.2f} microns')
    return norm_average, average_norm, diffs
