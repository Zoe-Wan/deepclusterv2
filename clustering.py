# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import faiss
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn

ImageFile.LOAD_TRUNCATED_IMAGES = True



class ReassignedDataset(data.Dataset):


    def __init__(self, image_indexes, pseudolabels, dataset):
        self.imgs = self.make_dataset(image_indexes, pseudolabels)
        self.dataset = dataset

    def make_dataset(self, image_indexes, pseudolabels):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):

            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((idx, pseudolabel))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        idx, pseudolabel = self.imgs[index]
        return self.dataset[idx][0], self.dataset[idx][1], pseudolabel

    def __len__(self):
        return len(self.imgs)


def preprocess_features(npdata, pca=256):
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)
    
    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata








class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def run_kmeans(self, x, nmb_clusters, ngpu, epoch):

        n_data, self.d = x.shape
        # print(x[0][0])
        # faiss implementation of k-means
        clus = faiss.Kmeans(self.d, nmb_clusters,niter=20,seed=epoch,gpu=ngpu)
        clus.train(x)
        _, I = clus.index.search(x, 1)
        return [int(n[0]) for n in I]

    def compute_features(self, dataloader, model, N, bs):
        print('Compute features')
        model.eval()
        features=[]
        for i, (input_tensor, _) in enumerate(dataloader):
            with torch.no_grad():
                input_var = torch.autograd.Variable(input_tensor.cuda())
                # todo
                aux = model(input_var,False).data.cpu().numpy()

                aux = aux.astype('float32')
                features.extend(aux)

        features = np.array(features)
        return features

    def cluster(self, data, ngpu,epoch):

        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data)

        # cluster the data
        I = self.run_kmeans(xb, self.k, ngpu,epoch)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        
    
    
    def cluster_assign(self, dataset):

        assert self.images_lists is not None
        pseudolabels = []
        image_indexes = []
        for cluster, images in enumerate(self.images_lists):
            image_indexes.extend(images)
            pseudolabels.extend([cluster] * len(images))

        return ReassignedDataset(image_indexes, pseudolabels, dataset)

    def arrange_clustering(self, images_lists):
        pseudolabels = []
        image_indexes = []
        for cluster, images in enumerate(images_lists):
            image_indexes.extend(images)
            pseudolabels.extend([cluster] * len(images))
        indexes = np.argsort(image_indexes)
        return np.asarray(pseudolabels)[indexes]



