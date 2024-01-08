import os
import shutil


def save_images(data, directory_name, save_time):
    for clas in ["happy", "sad"]:
        print(clas)
        image_names = data.list_files(f"data/{clas}/*", shuffle=False)
        for image_name in image_names.take(len(image_names)):
            imaga_name_path = image_name.numpy().decode("utf-8")
            directory = os.path.join("saved_data", f'{directory_name}{save_time}', clas)
            image_path = os.path.join(directory, os.path.basename(imaga_name_path))
            if not os.path.exists(directory):
                os.makedirs(directory)
            print(f'Source: {imaga_name_path}, dest: {image_path}')
            shutil.copy(imaga_name_path, image_path)


def split_data(data, save_model_time):
    data = data.map(lambda x, y: (x / 255, y))
    print(f'Available data: {len(data)}')

    train_size = int(len(data) * .7)  # 70% of the data
    val_size = int(len(data) * .2) + 1 # 20% of the data
    test_size = int(len(data) * .1) + 1  # 10% of the data

    print(f'Training data size: {train_size}')
    print(f'Validation data size: {val_size}')
    print(f'Test data size: {test_size}')

    train_data = data.take(train_size)
    val_data = data.skip(train_size).take(val_size)
    test_data = data.skip(train_size + val_size).take(test_size)

    save_images(test_data, "test", save_model_time)

    return train_data, val_data, test_data
