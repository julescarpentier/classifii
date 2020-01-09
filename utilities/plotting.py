import matplotlib.pyplot as plt


def plot_acc(fpath, history):
    """
    Plot training & validation accuracy values

    :param history: from model.fit
    :param fpath: output png file
    :return:
    """
    plt.figure()
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['acc'], '--')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(fpath)
    plt.close()


def plot_loss(fpath, history):
    """
    Plot training & validation loss values

    :param history: from model.fit
    :param fpath: output png file
    :return:
    """
    plt.figure()
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'], '--')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(fpath)
    plt.close()


def plot_compare_acc(fpath, legend, *histories):
    """
    Plot histories accuracies to be compared

    :param fpath:
    :param legend:
    :param histories:
    :return:
    """
    plt.figure()
    for history in histories:
        plt.plot(history.history['val_acc'])
    for history in histories:
        plt.plot(history.history['acc'], '--')
    plt.title('Compared accuracies')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(legend * 2, loc='upper left')
    plt.savefig(fpath)
    plt.close()


def plot_compare_loss(fpath, legend, *histories):
    """
    Plot histories losses to be compared

    :param fpath:
    :param legend:
    :param histories:
    :return:
    """
    plt.figure()
    for history in histories:
        plt.plot(history.history['val_loss'])
    for history in histories:
        plt.plot(history.history['loss'], '--')
    plt.title('Compared losses')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legend * 2, loc='upper left')
    plt.savefig(fpath)
    plt.close()
