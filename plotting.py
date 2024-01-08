from matplotlib import pyplot as plt


def plot_loaded_data(batch):
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])
    plt.show()


def plot_model_evaluation(hist):
    # Plot performance
    fig, axs = plt.subplots(2)
    axs[0].plot(hist.history['loss'], color='teal', label='loss')
    axs[0].plot(hist.history['val_loss'], color='orange', label='val_loss')
    axs[0].suptitle('Loss', fontsize=20)
    axs[0].legend(loc="upper left")

    axs[1].plot(hist.history['accuracy'], color='teal', label='accuracy')
    axs[1].plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    axs[1].suptitle('Accuracy', fontsize=20)
    axs[1].legend(loc="upper left")
    plt.show()