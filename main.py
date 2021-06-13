import os
import cv2
import itertools
import numpy as np
from glob import glob
from pylsd.lsd import lsd
from helpers_methods import angle_range, get_transform, order_points, filter_corner


class DocumentScanner:
    def __init__(self, img_paths, interactive=False):
        self.img_path = img_paths
        self.interactive = interactive
        self.rescaled_height = 500
        self.min_quad_area_ratio = 0.1

    def resize_image(self, img):
        h, w = img.shape[:2]
        ratio = h/self.rescaled_height
        dim = (int(w/ratio), 500)            # tuple(width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return img

    def is_valid_contour(self, cnt, img_w, img_h):
        return len(cnt) == 4 and cv2.contourArea(cnt) > img_w * img_h * self.min_quad_area_ratio

    @staticmethod
    def get_corners(edged_img):
        lines = lsd(edged_img)
        # lines is list of list with value of each list item is
        # [point1.x, point1.y, point2.x, point2.y, width]

        corners = []
        if lines is not None:
            horizontal_lines_canvas = np.zeros(edged_img.shape, dtype=np.uint8)
            vertical_lines_canvas = np.zeros(edged_img.shape, dtype=np.uint8)
            for line in lines:
                x1, y1, x2, y2, _ = line
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, edged_img.shape[1] - 1), y2),
                             255, 2)
                else:
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, edged_img.shape[0] - 1)),
                             255, 2)

            lines = []
            # find the horizontal lines (connected-components -> bounding boxes -> final lines)
            (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:5]

            horizontal_lines_canvas = np.zeros(edged_img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.min(contour[:, 0]) + 2
                max_x = np.max(contour[:, 0]) - 2
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                lines.append((min_x, left_y, max_x, right_y))
                cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 255, 1)
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))

            # find the vertical lines (connected-components -> bounding boxes -> final lines)
            (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:5]

            vertical_lines_canvas = np.zeros(edged_img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1]) + 2
                max_y = np.amax(contour[:, 1]) - 2
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                lines.append((top_x, min_y, bottom_x, max_y))
                cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 255, 1)
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))
            if corners:
                corners = filter_corner(corners)      # calling from helpers file
        return corners

    def get_contour(self, resized_image):
        img_h, img_w = resized_image.shape[:2]

        # convert into gray_scale
        gray_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.GaussianBlur(gray_img, (9, 9), 0)

        # dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel, iterations=3)

        edged = cv2.Canny(dilated, 0, 100)

        corners = self.get_corners(edged)

        approx_contours = []
        if len(corners) >= 4:
            quads = []
            for quad in itertools.combinations(corners, 4):    # quad is tuple of 4 point(tuple)
                points = order_points(np.array(quad))          # helpers_methods.py
                points = np.array([[p] for p in points], dtype='int32')
                quads.append(points)
            # top 5 quads with maximum area
            quads = sorted(quads, key=cv2.contourArea, reverse=True)[:10]
            # we can add angle range to remove outlier

            quads = sorted(quads, key=angle_range)  # helpers_methods.py

            approx_contours = list()
            for quad in quads:
                if self.is_valid_contour(quad, img_w, img_h):
                    # print("quad ", quad)
                    # shape of quad : [[[1,2]], [[3,4]], [[4,5]], [[6,7]]]
                    approx_contours.append(quad)
                    break
            print("app_con", approx_contours)
            # shape of approx contour is list of np.array([[[1,2]], [[3,4]], [[4,5]], [[6,7]]])
            if len(approx_contours) != 0:
                image = resized_image.copy()
                cv2.drawContours(image, approx_contours, -1, (20, 20, 255), 2)
                approx_contours2 = approx_contours[0]
                approx_contours2 = approx_contours2.reshape((approx_contours2.shape[0], approx_contours2.shape[2]))
                for p in approx_contours2:
                    image = cv2.circle(image, center=tuple(p), radius=5, color=255, thickness=1)
                cv2.imshow("img", image)
                cv2.waitKey(0)
                cv2.destroyWindow('img')

        # sometimes use direct will give better result
        (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for ct in cnts:
            approx = cv2.approxPolyDP(ct, 80, True)
            if self.is_valid_contour(approx, img_w, img_h):
                approx_contours.append(approx)
                break

        # If we did not find any valid contours, just use the whole image
        if not approx_contours:
            top_right = (img_w, 0)
            bottom_right = (img_w, img_h)
            bottom_left = (0, img_h)
            top_left = (0, 0)
            best_contour = np.array([[top_right], [bottom_right], [bottom_left], [top_left]])
        else:
            best_contour = min(approx_contours, key=cv2.contourArea)
        print(best_contour.shape)
        return best_contour.reshape(4, 2)

    def scan(self):
        for path in self.img_path:
            img = cv2.imread(path)
            assert (img is not None)
            # print(img.shape)
            cv2.imshow("image", img)
            cv2.waitKey(0)
            cv2.destroyWindow('image')
            ratio = img.shape[0] / self.rescaled_height
            original_img = img.copy()
            resized_img = self.resize_image(img)

            # get the contour of the document
            contour = self.get_contour(resized_img)
            # print("con", contour)

            # if self.interactive:
            #     contour = self.interactive_get_contour(contour, resized_img)

            transformed = get_transform(contour * ratio, original_img)  # helpers_methods.py
            cv2.imshow("trans", transformed)
            cv2.waitKey(0)
            cv2.destroyWindow('trans')

            # convert the warped image to grayscale
            gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)

            # sharpen image
            sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
            sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

            # apply adaptive threshold to get black and white effect
            thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
            cv2.imshow("transformed", thresh)
            cv2.waitKey(0)
            cv2.destroyWindow('transformed')
            print("done scanning for img: ", path)


if __name__ == '__main__':
    # video_capture = cv2.VideoCapture(0)
    # i = 0
    # while True:
    #     try:
    #         path_to_save = os.getcwd() + '\\data\\deep{0}.jpg'.format(str(i))
    #         _, frame1 = video_capture.read()
    #         # canvas = obj.detect(frame1)
    #         cv2.imshow('Video', frame1)
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord('s'):
    #             cv2.imwrite(path_to_save, frame1)
    #             sleep(2)
    #             i += 1
    #         if key == ord('q'):
    #             cv2.destroyWindow('Video')
    #             break
    #     except KeyboardInterrupt:
    #         video_capture.release()
    #         cv2.destroyAllWindows()
    # video_capture.release()

    paths = glob(os.getcwd() + r'\data\*jpg')
    obj = DocumentScanner(paths)
    obj.scan()
    cv2.destroyAllWindows()
