# -*- coding: utf-8 -*-
# @Time    : 2019/10/26 22:26
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : OpenCVAnalysis.py
# @Project : RoadExtraction
# @Software: PyCharm

import cv2
import numpy as np
from PyQt5.QtGui import QImage,  QColor
from DetectObjects.CircleSeed import CircleSeedNp
from Test.CircleSeedItem import CircleSeedItem
from DetectObjects.Utils import qimage2cvmat
from matplotlib import pyplot as plt


def show_grabcut_info(src_image: QImage, circle_seed_item: CircleSeedItem):
    """
    :param src_image:
    :param circle_seed_item:
    :return:
    """
    seed_path = circle_seed_item.path
    seed_rect = seed_path.boundingRect().toRect()

    seed_image = src_image.copy(seed_rect)
    cv_image = qimage2cvmat(seed_image)  # type: np.ndarray
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB)

    mask = np.zeros(cv_image.shape[:2], np.uint8)
    rect = (1, 1, seed_rect.width()-1, seed_rect.height()-1)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(cv_image, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) * mask2[:, : np.newaxis]

    # 显示原始图片
    cv2.imshow("origin image", cv_image)
    plt.imshow(img)
    plt.colorbar()
    plt.show()


def show_analysis_info(src_image: QImage, circle_seed_item: CircleSeedItem):

    """获取种子的Image"""
    seed_path = circle_seed_item.path
    seed_rect = seed_path.boundingRect().toRect()

    seed_image = src_image.copy(seed_rect)
    cv_image = qimage2cvmat(seed_image)  # type: np.ndarray
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB)

    # 图片二值化
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    _, bin_image = cv2.threshold(gray_image, 168, 255, cv2.THRESH_BINARY)

    print("src image shape: ", cv_image.shape)
    print("gray image shape: ", gray_image.shape)
    print("bin image shape: ", bin_image.shape)

    image = np.hstack((gray_image, bin_image))

    cv2.imshow("origin image", cv_image)
    cv2.imshow("Seed OpenCV Analysis", image)

    cv2.waitKey()
    cv2.destroyAllWindows()


def compare_tow_seed_of_spectral_info(image:np.ndarray, seed1: CircleSeedNp, seed2: CircleSeedNp):
    shape = image.shape
    print("seed1 光谱特征：", seed1.spectral_feature_vector)
    print("seed2 光谱特征：", seed2.spectral_feature_vector)
    print("局部交叉道路数目： ", np.intersect1d(seed1.get_pixels_position(shape), seed2.get_pixels_position(shape)).size)
    distance = np.linalg.norm(np.subtract(seed1.spectral_feature_vector, seed2.spectral_feature_vector))
    print("光谱特征距离：", distance)


def compare_to_seed(src_image: QImage, seed_item1: CircleSeedItem, seed_item2: CircleSeedItem):
    seed_image1 = src_image.copy(seed_item1.path.boundingRect().toRect())
    seed_image2 = src_image.copy(seed_item2.path.boundingRect().toRect())

    cv_image1 = qimage2cvmat(seed_image1)
    cv_image2 = qimage2cvmat(seed_image2)

    cv_image1 = cv2.cvtColor(cv_image1, cv2.COLOR_RGBA2RGB)  # type: np.ndarray
    cv_image2 = cv2.cvtColor(cv_image2, cv2.COLOR_RGBA2RGB)  # type: np.ndarray

    similarity_info = a_hash_similarity(cv_image1, cv_image2)
    print("相似度：", similarity_info)


def a_hash_similarity(image1: np.ndarray, image2: np.ndarray):
    hash_info1 = a_hash(image1)
    hash_info2 = a_hash(image2)

    diff_num = 0
    for index in range(64):
        if hash_info1[index] != hash_info2[index]:
            diff_num += 1

    return diff_num / 64


def a_hash(image: np.ndarray):
    img = image.copy()
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    s = 0

    for i in range(8):
        for j in range(8):
            s += img[i, j]

    avg = s / 64
    result = []
    for i in range(8):
        for j in range(8):
            if img[i, j] > avg:
                result.append(1)
            else:
                result.append(0)

    return result


def threshold(src_img: QImage, low: int, height: int, val) -> QImage:
    """
    :param src_img:
    :param low:
    :param height:
    :param val:
    :return:
    """
    gray_img = src_img.convertToFormat(QImage.Format_Grayscale8)
    res_img = QImage(src_img.width(), src_img.height(), QImage.Format_RGB888)

    # color = gray_img.pixelColor(364, 285)
    # print(color.red(), color.green(), color.blue())
    # color = gray_img.pixelColor(365, 285)
    # print(color.red(), color.green(), color.blue())
    # color = gray_img.pixelColor(365, 286)
    # print(color.red(), color.green(), color.blue())
    # color = gray_img.pixelColor(365, 287)
    # print(color.red(), color.green(), color.blue())
    # color = gray_img.pixelColor(366, 285)
    # print(color.red(), color.green(), color.blue())

    for y in range(0, src_img.height()):
        for x in range(0, src_img.width()):
            color = gray_img.pixelColor(x, y)
            if low < color.green() < height:
                res_img.setPixelColor(x, y, QColor(val, val, val))
            else:
                res_img.setPixelColor(x, y, QColor(0, 0, 0))

    return res_img


if __name__ == '__main__':
    # import time
    # import sys
    # img = cv2.imread("F:/RoadDetectionTestImg/4.png")
    # print(sys.getsizeof(img))
    # m = np.ones(img.shape)
    # print(sys.getsizeof(m))
    # n = np.ones(img.shape[:2], dtype=np.bool)
    # print(sys.getsizeof(n))
    # position = [1000, 400]
    # radius = 11
    # x_axis = np.arange(position[0] - radius, position[0] + radius, 1, np.int)
    # y_axis = np.arange(position[1] - radius, position[1] + radius, 1, np.int)
    # pixels_position = np.array(np.meshgrid(*[x_axis, y_axis])).T
    # pixels_position = pixels_position.reshape((pixels_position.shape[0] * pixels_position.shape[1], 2))
    # t = time.time()
    # pos = img[pixels_position[:, 1], pixels_position[:, 0]]
    # dt = time.time() - t
    # print("时间： ", dt)
    # print(len(pos))
    # print(pos)
    # print(polygon.count())

    from DetectObjects.Utils import get_circle_path
    position = [800, 680]
    radius = 12
    circle_path = get_circle_path(position, radius)
    control_pos_rect = circle_path.controlPointRect().toRect()
    bounding_rect = circle_path.boundingRect().toRect()
    print("control pos rect: ", control_pos_rect.topLeft(), " ", control_pos_rect.size())
    print("control pos rect: ", bounding_rect.topLeft(), " ", bounding_rect.size())
