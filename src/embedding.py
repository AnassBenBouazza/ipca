import time
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output
from sklearn.decomposition import PCA, IncrementalPCA
from interface import Run
from visualizer import show_components

class batch_PCA:

    """
    batch_PCA: compute PCA of a given image batch.
    """

    def __init__(self, batch):
        """
        """
        self.batch = batch
        self.n_images, self.nx, self.ny = batch.shape
        self.components, self.explained_variance_ratio_ = self._compute_components()

    def _compute_components(self):
        """
        """
        try:
            start_time = time.time()
            pca = PCA()
            pca.fit(self.batch.reshape(self.n_images, self.nx*self.ny))
            fit_time = time.time()
            print(f'PCA> fit in {(fit_time - start_time):.2f} s')
        except:
            print("PCA of batch did not succeed. Try run_PCA()")
        components = pca.transform(self.batch.reshape(self.n_images, self.nx*self.ny))
        explained_variance_ratio_ = pca.explained_variance_ratio_
        transform_time = time.time()
        print(f'PCA> components transformed in {(transform_time - fit_time):.2f} s')
        return components, explained_variance_ratio_

    def _compute_eigenimages(self):
        """
        """
        try:
            start_time = time.time()
            eigenimages = np.dot(self.components.T,
                                 self.batch.reshape(self.n_images, self.nx*self.ny))
            dot_time = time.time()
            print(f'PCA> eigenimages computed in {(dot_time - start_time):.2f} s')
        except:
            print("Error. Check that components were computed first.")
        return eigenimages.reshape(self.n_images, self.nx, self.ny)


class run_PCA:

    """
    """

    def __init__(self, expt, run_idx, det=None, image_type=None, batch_size=200):
        """
        """
        self.run = Run(expt, run_idx, det=det, image_type=image_type)
        self.n_components = 10
        self.batch_size = batch_size
        self.n_batches = self.run._n_images//self.batch_size

    def _compute_components(self):
        """
        """
        pca = IncrementalPCA(n_components=self.n_components)
        self.components = np.zeros((self.run._n_images,self.n_components))

        for i in tqdm(range(0, self.n_batches)):

            if i>0:
                show_components(self, n_sample=i*self.batch_size)

            start_time = time.time()
            images = self.run._retrieve_image_batch(batch_id=i,batch_size=self.batch_size)
            n_images, nx, ny = images.shape
            load_time = time.time()
            print(f'> batch loaded in {(load_time - start_time):.2f} s')

            pca.partial_fit(images.reshape((n_images, nx*ny)))
            fit_time = time.time()
            print(f'> partial fit in {(fit_time - load_time):.2f} s')

            self.components[i*self.batch_size:(i+1)*self.batch_size,:] = pca.transform(images.reshape((n_images, nx*ny)))
            self.explained_variance_ratio_ = pca.explained_variance_ratio_
            transform_time = time.time()
            print(f'> partial transform in {(transform_time - fit_time):.2f} s')

            end_time = time.time()
            print(f'total time = {(end_time - start_time):.2f} s')
            clear_output(wait=True)
