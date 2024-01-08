from data_preparation import remove_incorrect_images, load_data
from data_preprocessing import split_data
from gpu_setup import gpu_setup
from model import build_model, train_model, evaluate_model, test_model, save_model, load_model_from_path, test_model_dataset
from plotting import plot_model_evaluation, plot_loaded_data
from datetime import datetime
import os


def train(save_model_time):
    gpu_setup()
    remove_incorrect_images()
    data, batch = load_data('data')
    # plot_loaded_data(batch)
    train_data, val_data, test_data = split_data(data, save_model_time)
    model = build_model()
    hist, model = train_model(model, train_data, val_data)

    return model, test_data, hist


def train_test_model():
    save_model_time = datetime.now().timestamp()
    model, test_data, hist = train(save_model_time)
    plot_model_evaluation(hist)
    evaluate_model(model, test_data)
    save_model(model, f'happysadmodel{save_model_time}.h5')
    test_model_dataset(model, os.path.join("saved_data", f"test{save_model_time}"))
    # test_model(model, "sad_test3.jpg")


def load_test_model():
    save_model_time = "1704661267.149294"
    model = load_model_from_path(f'happysadmodel{save_model_time}.h5')
    test_model_dataset(model, os.path.join("saved_data", f"test{save_model_time}"))
    # test_model(model, "happy_test3.jpg")


if __name__ == '__main__':
    # train_test_model()
    load_test_model()
