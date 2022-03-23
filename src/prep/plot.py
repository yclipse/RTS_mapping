
import matplotlib.pyplot as plt


def plotim(**images):
    """
    PLot images in one row with title

    example:
    utils.plotim(rgb=test[...,:3], label=test[...,-1])

    """

    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def multiplots(image, label, plotsize=(3, 3)):
    '''plot images in subplot grids'''

    fig = plt.figure(figsize=(22, 22))
    for i in range(plotsize[0]*plotsize[1]):
        ax = fig.add_subplot(plotsize[0], plotsize[1], i+1, xticks=[], yticks=[])
        ax.contour(label[i])
        ax.imshow(image[i])
