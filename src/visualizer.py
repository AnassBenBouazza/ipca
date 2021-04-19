from matplotlib import pyplot as plt
import numpy as np

def show_image(image, title=None):
    vmin = np.mean(image)-3*np.std(image)
    vmax = np.mean(image)+3*np.std(image)
    fig, ax = plt.subplots(1,1,figsize=(3,3),dpi=180)
    if title is not None:
        ax.set_title(title)
    im = ax.imshow(image, vmin=vmin, vmax=vmax)
    fig.colorbar(im, orientation='vertical')
    plt.show()

def show_components(pca, n_sample=None):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(6,3),dpi=180)
    ax1.set_title('explained variance ratio')
    ax1.plot(pca.explained_variance_ratio_, 'o-')
    ax2.set_title('distribution in PC space')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    if n_sample is None:
        im2 = ax2.hexbin(pca.components[:,0], pca.components[:,1], mincnt=1,gridsize=100)
    else:
        im2 = ax2.hexbin(pca.components[:n_sample,0], pca.components[:n_sample,1], mincnt=1,gridsize=100)
    cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
