# -*- encoding: utf-8 -*-
# @File    : KMeanCluster.py
# @Time    : 2019/11/20 19:28
# @Author  : 一叶星羽
# @Email   : h0670131005@gmail.com
# @Software: PyCharm

import time
import numpy as np
import cv2
from sklearn.cluster import KMeans
from Core.Cluster.FeatureExtraction import feature_extraction
from Core.Cluster.FeatureExtraction1 import feature_extraction2


def k_mean_cluster(src_img: np.ndarray, road_feature_vector: np.ndarray = None):
    """
    获得每个像元的特征向量后，将光谱特征和纹理特征相似区域归类为同类区域，
    然后利用改进后的K-均值聚类算法进行分析，进一步进行数据处理，
    即对同类区域中的异类区域进行分割提取去除，以便获得最佳的图像道路分割结果.
    :param src_img: 原图像矩阵
    :param road_feature_vector: 用户采集的道路样本特征向量
    :return: 返回距离结果
    """
    st = time.time()
    img_feature_vector = feature_extraction2(src_img)
    print(img_feature_vector[-1])
    print("特征提取：", time.time() - st)

    if road_feature_vector is not None:
        feature_vector_distances = np.sqrt(np.sum(np.square(img_feature_vector - road_feature_vector), 1))
        non_road_feature_vector = img_feature_vector[np.argmax(feature_vector_distances), :]
        init_cluster_center = np.array((road_feature_vector, non_road_feature_vector))
        n_init = 1
    else:
        init_cluster_center = 'k-means++'
        n_init = 10
    road_model = KMeans(2, init=init_cluster_center, n_init=n_init)

    st = time.time()
    road_model.fit(img_feature_vector)
    print("均值聚类：", time.time() - st)

    labels = road_model.labels_.reshape(src_img.shape[0], src_img.shape[1]).astype(np.uint8)
    labels[labels == 1] = 255
    return labels


if __name__ == '__main__':
    src_image = cv2.imread("../../TestImg/6.png")
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    res_image = k_mean_cluster(src_image)

    print(res_image.shape)
    cv2.imshow("cluster result", res_image)
    cv2.waitKey(0)
    cv2.destroyWindow("image")
