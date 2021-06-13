import cv2
import math
import numpy as np
import pandas as pd


def angle_range(quad):
    def angle_bet_vectors_deg(x, y):
        return np.degrees(math.acos(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))))

    def get_angle(p1, p2, p3):
        a = np.radians(np.array(p1))
        b = np.radians(np.array(p2))
        c = np.radians(np.array(p3))
        a_vec = a - b
        c_vec = c - b
        return angle_bet_vectors_deg(a_vec, c_vec)

    tl, tr, br, bl = quad
    ura = get_angle(tl[0], tr[0], br[0])
    ula = get_angle(bl[0], tl[0], tr[0])
    lra = get_angle(tr[0], br[0], bl[0])
    lla = get_angle(br[0], bl[0], tl[0])

    angles = [ula, ura, lra, lla]

    return np.ptp(angles)


def order_points(quad):                                     # # [[6, 2], [4, 1], [3, 9], [8, 4]]
    quad = sorted(quad, key=lambda x: x[0])                 # [[3, 9], [4, 1], [6, 2], [8,4]]
    left_part = quad[:2]                                    # [[3, 9], [4, 1]]
    right_part = quad[2:]                                   # [[6, 2], [8,4]]
    tl, bl = sorted(left_part, key=lambda x: x[1])          # tl ,bl = [4, 1], [3, 9]
    tr, br = sorted(right_part, key=lambda x: x[1])         # tr, br = [6, 2], [8, 4]
    return np.array([tl, tr, br, bl], dtype="float32")


def filter_corner(corners):
    final_corners = [corners[0]]                            # final list of corners
    # min distance between two point is 20, if below 20 then filter
    # np.linalg.norm(pt1, pt2) is used to calculate the euclidean distance between pt1 and pt2
    for corner in corners[1:]:
        d = [True if (np.linalg.norm(np.array(cor) - np.array(corner)) >= 20) else False for cor in final_corners]
        if all(d):
            final_corners.append(corner)
    return final_corners


def get_transform(contr, orig_img):
    contr = order_points(contr)
    (tl, tr, br, bl) = contr
    # width of the image
    width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_img = max(int(width1), int(width2))

    # height of image
    height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height_img = max(int(height1), int(height2))

    point_of_img = np.array([[0, 0],
                            [width_img, 0],
                            [width_img, height_img],
                            [0, height_img]], dtype="float32")

    method = cv2.getPerspectiveTransform(contr, dst=point_of_img)
    warped = cv2.warpPerspective(orig_img, method, (width_img, height_img))
    return warped
