import numpy as np
import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class Cifar10(object):
    def __init__(self, filepath, noisy=False):
        self.__train_files = [os.path.join(filepath, 'data_batch_'+str(i)) for i in [1, 2, 3, 4, 5]]
        self.__test_file = os.path.join(filepath, 'test_batch')

        self.__train_imgs = None
        self.__train_labels = None
        self.__test_imgs = None
        self.__test_labels = None

        self.__index = 0
        self.__constuctor_train__()
        self.__data_augment()
        if noisy:
            self.__data_noisy()
        self.__constructor_test__()

    def next_batch(self, num):
        assert isinstance(num, int) and num > 0, "batch请指定为正整数"
        if self.__index < (50000 - 1 - self.__index):
            self.__index += num
            return self.__train_imgs[self.__index - num:self.__index], \
                   self.__train_labels[self.__index - num:self.__index]
        else:
            self.__index = 0
            return self.__train_imgs[-num:], self.__train_labels[-num:]

    @property
    def test_imgs(self):
        return self.__test_imgs

    @property
    def test_labels(self):
        return self.__test_labels

    def __constuctor_train__(self):
        self.__train_imgs = np.zeros(shape=(5*10000, 32, 32, 3), dtype=float)
        self.__train_labels = np.zeros(shape=(5*10000, 10), dtype=int)
        for num in range(len(self.__train_files)):
            train_data = unpickle(self.__train_files[num])
            data = train_data[b"data"]
            # 将data归一到[-1, 1]
            data = (data - 127.0) / 127.0
            label = train_data[b'labels']

            # 将label转换成one-hot格式
            for i in range(len(label)):
                self.__train_labels[num*10000+i, label[i]] = 1

            # 将data转换成[batch, high, width, channles]格式
            '''
            data:一个10000*3072的numpy数组，数据类型是无符号整形uint8。
            这个数组的每一行存储了32*32大小的彩色图像（32*32*3通道=3072）。
            前1024个数是red通道，然后分别是green,blue
            '''
            r = data[:, :1024]
            g = data[:, 1024:2048]
            b = data[:, 2048:]
            for i in range(len(data)):
                self.__train_imgs[num*10000+i, :, :, 0] = np.reshape(r[i], newshape=(32, 32))
                self.__train_imgs[num*10000+i, :, :, 1] = np.reshape(g[i], newshape=(32, 32))
                self.__train_imgs[num*10000+i, :, :, 2] = np.reshape(b[i], newshape=(32, 32))

    def __data_augment(self):
        imgs = self.__train_imgs
        labels = self.__train_labels
        imgs = imgs[:, :, ::-1, :]
        self.__train_imgs = np.concatenate([self.__train_imgs, imgs], axis=0)
        self.__train_labels = np.concatenate([labels, labels], axis=0)
        # random
        self.__random()

    def __data_noisy(self):
        noisy_img = np.random.standard_normal((len(self.__train_labels), 32, 32, 3))/20
        imgs = self.__train_imgs + noisy_img
        self.__train_imgs = np.concatenate([self.__train_imgs, imgs], axis=0)
        self.__train_labels = np.concatenate([self.__train_labels, self.__train_labels], axis=0)
        self.__random()

    def __random(self):
        permutation = np.random.permutation(self.__train_labels.shape[0])
        self.__train_imgs = self.__train_imgs[permutation, :, :, :]
        self.__train_labels = self.__train_labels[permutation, :]

    def __constructor_test__(self):
        self.__test_imgs = np.zeros(shape=(10000, 32, 32, 3), dtype=float)
        self.__test_labels = np.zeros(shape=(10000, 10), dtype=int)

        test_data = unpickle(self.__test_file)
        data = test_data[b"data"]
        # 将data归一到[-1, 1]
        data = (data - 127.0) / 127.0
        label = test_data[b'labels']

        # 将label转换成one-hot格式
        for i in range(len(label)):
            self.__test_labels[i, label[i]] = 1

        r = data[:, :1024]
        g = data[:, 1024:2048]
        b = data[:, 2048:]
        for i in range(len(data)):
            self.__test_imgs[i, :, :, 0] = np.reshape(r[i], newshape=(32, 32))
            self.__test_imgs[i, :, :, 1] = np.reshape(g[i], newshape=(32, 32))
            self.__test_imgs[i, :, :, 2] = np.reshape(b[i], newshape=(32, 32))


if __name__ == "__main__":
    # filepath = "/home/yangdehe/Wangmeng/remote_cifar100/cifar-100-python/cifar-100-python/"
    filepath = "D:\\data\\cifar-10-python\\cifar-10-batches-py"
    cifar10 = Cifar10(filepath)

