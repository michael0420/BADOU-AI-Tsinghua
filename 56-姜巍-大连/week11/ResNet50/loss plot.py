from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import numpy as np


def polygon_under_graph(xx, yy):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(xx[0], 0.), *zip(xx, yy), (xx[-1], 0.)]


if __name__ == '__main__':
    # 导入训练日志原始数据
    loss_by_mini_batches = [0.73026, 0.69471, 0.71476, 0.68118, 0.70628, 0.71672, 0.72101, 0.66136, 0.63982, 0.67456,
                            0.65856, 0.64144, 0.66133, 0.64138, 0.64162, 0.64587, 0.62394, 0.62694, 0.64327, 0.63127,
                            0.64836, 0.65862, 0.63179, 0.62356, 0.63470, 0.62085, 0.61112, 0.60423, 0.60109, 0.59027,
                            0.58368, 0.57242, 0.54886, 0.57976, 0.57755, 0.55217, 0.54305, 0.55332, 0.53085, 0.52547,
                            0.49721, 0.51434, 0.54034, 0.47676, 0.51092, 0.53504, 0.55247, 0.54225, 0.49326, 0.48985,
                            0.50359, 0.48786, 0.48292, 0.46952, 0.45875, 0.45593, 0.46313, 0.47903, 0.52883, 0.47431,
                            0.48439, 0.43713, 0.41758, 0.42255, 0.43504, 0.41569, 0.45061, 0.41344, 0.41767, 0.39625,
                            0.41022, 0.38170, 0.37968, 0.38193, 0.37674, 0.39423, 0.41398, 0.41144, 0.36543, 0.38500,
                            0.34694, 0.38779, 0.35370, 0.33973, 0.36681, 0.34986, 0.37342, 0.35358, 0.37168, 0.35762,
                            0.35751, 0.33637, 0.32150, 0.34678, 0.36074, 0.34670, 0.32726, 0.29930, 0.33670, 0.33499,
                            0.34530, 0.34072, 0.29580, 0.31753, 0.29827, 0.30160, 0.29214, 0.27001, 0.30236, 0.30049,
                            0.28410, 0.25741, 0.24485, 0.29171, 0.30821, 0.26390, 0.27994, 0.28421, 0.31000, 0.31747,
                            0.24677, 0.20027, 0.22755, 0.22067, 0.21060, 0.27197, 0.28656, 0.29137, 0.25082, 0.24332,
                            0.23169, 0.20405, 0.18308, 0.18672, 0.22189, 0.20894, 0.22703, 0.26801, 0.25333, 0.25177,
                            0.20582, 0.17076, 0.15945, 0.15960, 0.17736, 0.18059, 0.20541, 0.19285, 0.19940, 0.22426,
                            0.15747, 0.12856, 0.14789, 0.18087, 0.17001, 0.17783, 0.16311, 0.19759, 0.19853, 0.17485,
                            0.13637, 0.10347, 0.09009, 0.12812, 0.10768, 0.11573, 0.14468, 0.16030, 0.16074, 0.17889,
                            0.12300, 0.11174, 0.11252, 0.11194, 0.12016, 0.10541, 0.15150, 0.11509, 0.10286, 0.12810,
                            0.07654, 0.08562, 0.09902, 0.07149, 0.07917, 0.07375, 0.07232, 0.09453, 0.11125, 0.13463,
                            0.09307, 0.06348, 0.04786, 0.05615, 0.07786, 0.06798, 0.07424, 0.08200, 0.09259, 0.13963
                            ]

    # 将每20个mini-batches一采样输出的loss变成二维(行代表epoch，列代表采样次数)
    loss_by_mini_batches = np.array(loss_by_mini_batches).reshape((10, 20)).T

    # 作图部分，参考matplotlib官网教程
    plt.rcParams['font.sans-serif'] = ['simhei']
    ax = plt.figure(figsize=(12, 9), dpi=600).add_subplot(projection='3d')

    x = np.linspace(1, 20, 20, dtype=int)
    lambdas = range(10)

    # verts[i] is a list of (x, y) pairs defining polygon i.
    verts = [polygon_under_graph(x, loss_by_mini_batches[:, l]) for l in lambdas]
    facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))

    poly = PolyCollection(verts, facecolors=facecolors, alpha=0.7)
    ax.add_collection3d(poly, zs=lambdas, zdir='y')

    ax.set(xlim=(0, 20), ylim=(0, 10), zlim=(0, 0.75),
           xlabel='Epoch', ylabel=r'$\lambda$ * 15 mini-batches', zlabel='loss')

    ax.set_title("Loss随训练过程的变化趋势图")
    ax.view_init(30, 135)
    plt.show()
