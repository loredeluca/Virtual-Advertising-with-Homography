import cv2
import cv2.aruco as aruco
import numpy as np


if __name__ == '__main__':
    # Read source image
    im_src = cv2.imread("adv.jpg")
    size = im_src.shape

    # Read background video
    cap = cv2.VideoCapture('files/quadro+markers.MOV')

    vid_writer = cv2.VideoWriter("result_Hw4MV.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 28,
                                (round(2 * cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cv2.waitKey(1) < 0:
        try:
            ret, frame = cap.read()
            if not ret:
                break

            # Initialize aruco detector
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
            parameters = aruco.DetectorParameters_create()
            corners, ids, _  = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

            index = np.squeeze(np.where(ids == 25))
            refPt1 = np.squeeze(corners[index[0]])[1]

            index = np.squeeze(np.where(ids == 33))
            refPt2 = np.squeeze(corners[index[0]])[2]

            distance = np.linalg.norm(refPt1 - refPt2)

            scalingFac = 0.02
            pts_dst = [[refPt1[0] - round(scalingFac * distance), refPt1[1] - round(scalingFac * distance)]]
            pts_dst = pts_dst + [[refPt2[0] + round(scalingFac * distance), refPt2[1] - round(scalingFac * distance)]]

            index = np.squeeze(np.where(ids == 30))
            refPt3 = np.squeeze(corners[index[0]])[0]

            pts_dst = pts_dst + [[refPt3[0] + round(scalingFac * distance), refPt3[1] + round(scalingFac * distance)]]

            index = np.squeeze(np.where(ids == 23))
            refPt4 = np.squeeze(corners[index[0]])[0]

            pts_dst = pts_dst + [[refPt4[0] - round(scalingFac * distance), refPt4[1] + round(scalingFac * distance)]]
            pts_src = [[0, 0], [im_src.shape[1], 0], [im_src.shape[1], im_src.shape[0]], [0, im_src.shape[0]]]

            pts_src_m = np.asarray(pts_src)
            pts_dst_m = np.asarray(pts_dst)

            # Calculate Homography between source and destination points
            h, status = cv2.findHomography(pts_src_m, pts_dst_m)
            # Warp source image
            frame_tmp = cv2.warpPerspective(im_src, h, (frame.shape[1], frame.shape[0]))
            # Black out polygonal area in destination image
            mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32([pts_dst_m]), (255, 255, 255), cv2.LINE_AA)
            # Erode the mask to not copy the boundary effects from the warping
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.erode(mask, element, iterations=3)

            # Copy the mask into 3 channels.
            frame_tmp = frame_tmp.astype(float)
            mask3 = np.zeros_like(frame_tmp)
            for i in range(0, 3):
                mask3[:, :, i] = mask / 255

            # Copy the warped image into the original frame in the mask region.
            warped_image_masked = cv2.multiply(frame_tmp, mask3)
            frame_masked = cv2.multiply(frame.astype(float), 1 - mask3)
            output = cv2.add(warped_image_masked, frame_masked)

            # Showing the original image and the new output image side by side
            concatenatedOutput = cv2.hconcat([frame.astype(float), output])

            cv2.imshow("AR using Aruco markers", concatenatedOutput.astype(np.uint8))
            vid_writer.write(concatenatedOutput.astype(np.uint8))

        except Exception as e:
            print(e)

    cap.release()
    vid_writer.release()
    print('Video writer released..')

    cv2.destroyAllWindows()
