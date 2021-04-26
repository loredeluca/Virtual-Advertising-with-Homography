
import cv2
import numpy as np
from utils import get_four_points


def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

if __name__ == '__main__':
    # Read in the image.
    im_src = cv2.imread("files/quadro.jpg")
    overlay = cv2.imread("files/adv.jpg")
    size1 = im_src.shape
    s1 = size1[0]
    s2 = size1[1]
    size1 = (s2, s1)
    print(size1)

    # Destination image
    #size = (600, 400, 3)
    size = (4032, 3024)

    im_dst = np.zeros(size, np.uint8)

    pts_dst = np.array(
        [
            [0, 0],
            [size[0] - 1, 0],
            [size[0] - 1, size[1] - 1],
            [0, size[1] - 1]
        ], dtype=float
    )

    print(" ")
    '''
        Click on the four corners of the book -- top left first and
        bottom left last -- and then hit ENTER
        '''

    # Show image and wait for 4 clicks.
    cv2.imshow("Image", im_src)
    pts_src = get_four_points(im_src)

    # Calculate the homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination
    im_dst = cv2.warpPerspective(im_src, h, size[0:2])
    #trovo la matrice inversa
    h_inv=np.linalg.inv(h)

    #metto l'immagine sopra l'img rettificata

    #cv2.imshow("Image", im_dst)
    #pp = get_four_points(im_dst)
    #cv2.fillConvexPoly(im_dst, pp.astype(int), 0, 16)
    #backg = im_dst#+overlay#overlay_transparent(im_dst, overlay, 100,100)

    width = 3600
    height = 2600
    dim = (width, height)

    # resize image
    r_img = cv2.resize(overlay, dim, interpolation=cv2.INTER_AREA)
    hhh, www, _ = r_img.shape

    # load background image as grayscale
    hh, ww, _ = im_dst.shape

    # compute xoff and yoff for placement of upper left corner of resized image
    yoff = round((hh - hhh) / 2)
    xoff = round((ww - www) / 2)

    # use numpy indexing to place the resized image in the center of background image
    result = im_dst.copy()
    result[yoff:yoff + hhh, xoff:xoff + www] = r_img

    # faccio il procedimento inverso ottenendo il risultato finale
    im2 = cv2.warpPerspective(result, h_inv, size1[0:2])


    # Show output
    cv2.imshow("Image", im_dst)
    cv2.imshow("Image2", im2)
    cv2.imshow("img3", result)
    cv2.imwrite("result1_Correct Prospective.jpg", im_dst)
    cv2.imwrite("result2_Correct Prospective.jpg", im2)
    cv2.imwrite("result3_Correct Prospective.jpg", result)
    cv2.waitKey(0)

