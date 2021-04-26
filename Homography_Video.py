#from __future__ import print_function
import sys
import cv2
import numpy as np

bboxes = []
pts_dst = []
tmp_points = []


if __name__ == '__main__':
    # Read source image
    img_src = cv2.imread('files/adv.jpg')
    size = img_src.shape

    # Read background video
    #cap = cv2.VideoCapture("Uffizzi.mp4")
    cap = cv2.VideoCapture("files/videoQuadro.MOV")

    vid_writer = cv2.VideoWriter("result_HV.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 28, #(int(cap.get(3)),int(cap.get(4))))
                                (round(2 * cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Read first frame
    success, frame = cap.read()
    if not success:
        print('Failed to read video')
        sys.exit(1)

    # Get destination points
    while True:
        print('Center 4 points in the bounding box, in order (tl,tr,br,bl)')
        bbox = cv2.selectROI('MultiTracker', frame)
        bboxes.append(bbox)
        if len(bboxes) == 4:
            break
    print('Selected bounding boxes {}'.format(bboxes))

    # Create MultiTracker object
    multiTracker = cv2.legacy.MultiTracker_create()
    # Initialize MultiTracker
    for bbox in bboxes:
        multiTracker.add(cv2.legacy.TrackerCSRT_create(), frame, bbox)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get updated location of points in subsequent frames
        success, boxes = multiTracker.update(frame)

        # Get destination points in subsequent frames
        for i, newbox in enumerate(boxes):
            if i == 0:
                tmp_points = []
            x = int(newbox[0] + newbox[2] / 2)
            y = int(newbox[1] + newbox[3] / 2)
            # Views tracking of points
            # cv2.circle(frame, (x, y), radius=5, color=(255, 255, 255), thickness=-1)
            tmp_points.append((x, y))

        im_dst = frame
        pts_dst = np.array(tmp_points)
        pts_src = np.array([[0, 0], [size[1] - 1, 0], [size[1] - 1, size[0] - 1], [0, size[0] - 1]], dtype=float)

        # Calculate Homography between source and destination points
        h, status = cv2.findHomography(pts_src, pts_dst)
        # Warp source image
        frame_tmp = cv2.warpPerspective(img_src.copy(), h, (frame.shape[1], frame.shape[0]))
        frame_copy = frame.copy()
        # Black out polygonal area in destination image
        cv2.fillConvexPoly(frame, pts_dst.astype(int), 0, 16)
        # Add warped source image to destination frame
        output = cv2.add(frame, frame_tmp)

        concatenatedOutput = cv2.hconcat([frame_copy, output])
        # Display frame
        cv2.imshow('MultiTracker', output)

        vid_writer.write(concatenatedOutput.astype(np.uint8))#frame)

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    vid_writer.release()
    print('Video writer released..')

    cv2.destroyAllWindows()
