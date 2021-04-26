import cv2
import cv2.aruco as aruco
import numpy as np

if __name__ == '__main__':
    # Read source image
    im_src = cv2.imread('files/adv.jpg')
    size = im_src.shape

    # Read background video
    cap = cv2.VideoCapture('files/quadro+marker.MOV')

    vid_writer = cv2.VideoWriter("result_HwMV.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 28,# (int(cap.get(3)),int(cap.get(4))))
                                 (round(2 * cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # FOR RESIZE IMG
        # scale_percent = 100  # percent of original size
        # width = int(frame.shape[1] * scale_percent / 100)
        # height = int(frame.shape[0] * scale_percent / 100)

        # dim = (width, height)
        # resize image
        # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        output = frame

        # Initialize aruco detector
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        # Get destination points in subsequent frames
        if np.all(ids != None):
            for c in corners:
                x1 = (c[0][0][0], c[0][0][1])
                x2 = (c[0][1][0], c[0][1][1])
                x3 = (c[0][2][0], c[0][2][1])
                x4 = (c[0][3][0], c[0][3][1])

                im_dst = frame
                pts_dst = np.array([x1, x2, x3, x4])
                pts_src = np.array([[0, 0], [size[1] - 1, 0], [size[1] - 1, size[0] - 1], [0, size[0] - 1]], dtype=float)

                # Calculate Homography between source and destination points
                h, status = cv2.findHomography(pts_src, pts_dst)
                # Warp source image
                frame_tmp = cv2.warpPerspective(im_src.copy(), h, (frame.shape[1], frame.shape[0]))
                frame_copy = frame.copy()
                # Black out polygonal area in destination image
                cv2.fillConvexPoly(output, pts_dst.astype(int), 0, 16)
                # Add warped source image to destination frame
                output = cv2.add(output, frame_tmp)

                concatenatedOutput = cv2.hconcat([frame_copy, output])
                # Display frame
                cv2.imshow('frame', output)
                vid_writer.write(concatenatedOutput.astype(np.uint8))  # frame)
        else:
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    vid_writer.release()
    print('Video writer released..')

    cv2.destroyAllWindows()
