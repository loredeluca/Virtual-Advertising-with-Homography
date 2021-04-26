from operator import itemgetter
from utils import *

memo = []
if __name__ == '__main__':

    img_src = cv2.imread('files/adv.jpg')
    size = img_src.shape

    cap = cv2.VideoCapture("files/foglio.MOV")

    vid_writer = cv2.VideoWriter("result_lines.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 28, #(int(cap.get(3)),int(cap.get(4))))
                                (round(2 * cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    prova = []
    frame_number=0

    while cap.isOpened():
        ret, orig_frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture("c1.mp4")
            continue
        frame_number+=1
        print(frame_number)

        frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_yellow = np.array([0, 128, 0])
        up_yellow = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, low_yellow, up_yellow)
        edges = cv2.Canny(mask, 75, 150)


        lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)  # , maxLineGap=50) #80
        #lines2 = cv2.HoughLines(edges, 1, np.pi / 180, 100)  # , maxLineGap=50) #220

        # utils
        tmp_dst = []
        tmp_sort = []
        new_pts_dst = []

        Hp = []
        Hp.append([(1, 0), (0, 0)])
        Hp.append([(1, 0), (0, 0)])
        Vp = []
        Vp.append([(1, 0), (0, 0)])
        Vp.append([(1, 0), (0, 0)])

        if lines is not None:
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))

                    m=(y2-y1)/(x2-x1)

                    if m > 0:
                        q = y1 - (m * x1)
                        if q > 200:
                            Vp[0] = [(x1, y1), (x2, y2)]
                        else:
                            Vp[1] = [(x1, y1), (x2, y2)]
                    else:
                        q = y1 - (m * x1)
                        if q > 2700:
                            Hp[0] = [(x1, y1), (x2, y2)]
                        else:
                            Hp[1] = [(x1, y1), (x2, y2)]

        for lh in Hp:
            for lv in Vp:
                Ax1 = lh[0][0]
                Ay1 = lh[0][1]
                Ax2 = lh[1][0]
                Ay2 = lh[1][1]

                Bx1 = lv[0][0]
                By1 = lv[0][1]
                Bx2 = lv[1][0]
                By2 = lv[1][1]

                x, y = line_intersection(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2)

                if x > 0 and y > 0:
                    tmp_dst.append([int(x),int(y)])
                    tmp_dst = remove_duplicates(tmp_dst)

        tmp_dst = np.array(tmp_dst)
        tmp_dst = sorted(tmp_dst, key=itemgetter(0))

        #punti tmp
        p1 = tmp_dst[0]
        p2 = tmp_dst[1]
        p3 = tmp_dst[2]
        p4 = tmp_dst[3]
        print(p1, p2, p3, p4)

        tmp_sort.append(p2)
        tmp_sort.append(p4)
        tmp_sort.append(p3)
        tmp_sort.append(p1)

        # --
        p_d = np.array(tmp_sort)

        # retta orizzontale in basso
        distance0 = 90000

        m0, q0 = create_line(p_d[2][0], p_d[2][1], p_d[3][0], p_d[3][1])
        p_left = (p_d[2][0] - dx(distance0, m0), p_d[2][1] - dy(distance0, m0))  # invertendo + e - l'img è esterna o interna
        p_right = (p_d[3][0] + dx(distance0, m0), p_d[3][1] + dy(distance0, m0))

        # cv2.circle(frame, (int(p_left[0]), int(p_left[1])), 3, (0, 0, 255), 5, 16)
        # cv2.circle(frame, (int(p_right[0]), int(p_right[1])), 3, (0, 0, 255), 5, 16)

        distance = 40000

        # retta verticale 1
        m1, q1 = create_line(p_d[0][0], p_d[0][1], p_d[3][0], p_d[3][1])

        # retta verticale 2
        m, q = create_line(p_d[1][0], p_d[1][1], p_d[2][0], p_d[2][1])

        d1 = 1000

        pp1 = (p_right[0] - dx(d1, m1), p_right[1] - dy(d1, m1))
        pp2 = (p_left[0] - dx(d1, m), p_left[1] - dy(d1, m))
        pp3 = (p_left[0] - dx(distance, m), p_left[1] - dy(distance, m))
        pp4 = (p_right[0] - dx(distance, m1), p_right[1] - dy(distance, m1))
        # other_possible_point_b = (A[0] - dx(distance, m), A[1] - dy(distance, m))  # going the other way

        new_pts_dst.append([int(pp1[0]), int(pp1[1])])
        new_pts_dst.append([int(pp2[0]), int(pp2[1])])
        new_pts_dst.append([int(pp3[0]), int(pp3[1])])
        new_pts_dst.append([int(pp4[0]), int(pp4[1])])

        print("NPD")
        print(new_pts_dst)

        new_pts_dst = np.vstack(new_pts_dst).astype(float)
        # --

        # new_pts_dst = np.vstack(tmp_sort).astype(float)

        # Processo di omorgafia
        pts_src = np.array([[0, 0], [size[1] - 1, 0], [size[1] - 1, size[0] - 1], [0, size[0] - 1]], dtype=float)
        # Calculate Homography between source and destination points
        h, status = cv2.findHomography(pts_src, new_pts_dst)
        # Warp source image
        frame_tmp = cv2.warpPerspective(img_src.copy(), h, (frame.shape[1], frame.shape[0]))
        frame_copy = frame.copy()
        # Black out polygonal area in destination image
        cv2.fillConvexPoly(frame, new_pts_dst.astype(int), 0, 16)
        # Add warped source image to destination frame
        output = cv2.add(frame, frame_tmp)

        concatenatedOutput = cv2.hconcat([frame_copy, output])

        # Display frame
        cv2.imshow("frame", output) #frame
        #cv2.waitKey(00) == ord('k')
        #cv2.imshow("edges", edges)

        vid_writer.write(concatenatedOutput.astype(np.uint8))  # frame)

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    vid_writer.release() #per salvare il video
    print('Video writer released..')

    cv2.destroyAllWindows()
