import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from model import FCN
import torch
import numpy as np


def PCA_scatter(embedding_data):
    # 使用PCA降维度
    pca = PCA(n_components=2)
    embedding_2d = pca.fit_transform(embedding_data)

    # 创建散点图
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1])

    # 添加标签
    # word_list = [i for i in range(256)]
    # for i, word in enumerate(word_list):
    #     if i%5==0:
    #         plt.annotate(word, (embedding_2d[i, 0], embedding_2d[i, 1]))

    # 显示图形
    plt.show()


def TSNE_scatter(embedding_data):
    # 使用t-SNE降维到2维
    tsne = TSNE(n_components=2, random_state=42)
    embedding_2d = tsne.fit_transform(embedding_data)

    # 创建散点图
    labels = [i for i in range(256)]
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels)

    # 添加标签
    for i, label in enumerate(labels):
        if i % 5 == 0:  # 每5个数据点制定一个标识 防止标识之间重叠
            plt.annotate(label, (embedding_2d[i, 0], embedding_2d[i, 1]))

    # 显示图形
    plt.show()


def TSNE_scatter_3D(embedding_data):
    # 使用t-SNE降维到3维
    tsne = TSNE(n_components=3, random_state=42)
    embedding_3d = tsne.fit_transform(embedding_data)

    # 创建3D散点图
    labels = [i for i in range(256)]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1],
               embedding_3d[:, 2], c=labels)

    # 添加标签
    # for i, label in enumerate(labels):
    #     if i % 5 == 0:
    #         ax.text(embedding_3d[i, 0], embedding_3d[i, 1],
    #                 embedding_3d[i, 2], str(label))

    # 显示图形
    plt.show()


def hot_plot(embedding_data):
    # 创建热力图  cmap 控制颜色 cool hot viridis
    plt.imshow(embedding_data, cmap='cool', aspect='auto')

    # 添加颜色条
    plt.colorbar()

    # 显示图形
    plt.show()


def plot(embedding_data):
    # 创建一个新的图形
    fig = plt.figure(figsize=(10, 6))

    # 绘制平行坐标图
    for i in range(len(embedding_data)):
        # label参数 label=f"Sample {i+1}"
        plt.plot(range(embedding_data.shape[1]), embedding_data[i])

    # 添加坐标轴标签
    plt.xlabel('Dimensions')
    plt.ylabel('Values')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()


if __name__ == '__main__':
    model = FCN(embedding_dim=32, p=0.2, is_pe=False,
                is_wordEmbedding=True).cuda()
    model.load_state_dict(torch.load("model\FCN_2_24_bestModel.pth"))
    model.eval()
    raw_model = FCN(embedding_dim=32, p=0.2, is_pe=False,
                    is_wordEmbedding=True)
    raw_embedding_data = raw_model.embedding.weight.data.cpu().numpy()
    embedding_data = model.embedding.weight.data.cpu().numpy()

    one_hot = np.eye(256, dtype=np.float32)

    print("One-hot编码: ", one_hot)
    print("未训练的词嵌入: ", raw_embedding_data)
    print("经过训练后的词嵌入: ", embedding_data)

    PCA_scatter(embedding_data=one_hot)
    PCA_scatter(embedding_data=raw_embedding_data)
    PCA_scatter(embedding_data=embedding_data)

    TSNE_scatter(embedding_data=one_hot)
    TSNE_scatter(embedding_data=raw_embedding_data)
    TSNE_scatter(embedding_data=embedding_data)
    TSNE_scatter_3D(embedding_data=embedding_data)

    hot_plot(embedding_data=one_hot)
    hot_plot(embedding_data=raw_embedding_data)
    hot_plot(embedding_data=embedding_data)

    plot(embedding_data=one_hot)
    plot(embedding_data=raw_embedding_data)
    plot(embedding_data=embedding_data)
