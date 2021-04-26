from utils import *

new_pts_dst = []
new_p = []

if __name__ == '__main__':
    # Read source image
    img_src = cv2.imread('files/adv.jpg')
    size = img_src.shape

    pts_src = np.array([[0, 0], [size[1] - 1, 0], [size[1] - 1, size[0] - 1], [0, size[0] - 1]], dtype=float)

    # Read background image
    img_dst = cv2.imread('files/quadro.jpg')

    # Get destination points
    print('Click on four corners in order (tl,tr,br,bl), then press ENTER')
    pts_dst = get_four_points(img_dst)

    # Calculate Homography between source and destination points
    h, status = cv2.findHomography(pts_src, pts_dst)
    # Warp source image
    img_tmp = cv2.warpPerspective(img_src, h, (img_dst.shape[1], img_dst.shape[0]))
    # Black out polygonal area in destination image
    cv2.fillConvexPoly(img_dst, pts_dst.astype(int), 0, 16)
    # Add warped source image to destination image
    img_dst = img_dst + img_tmp

    # Display image
    cv2.imshow("Image", img_dst)
    cv2.imwrite("result.jpg", img_dst)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
