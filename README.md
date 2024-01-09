# Setup and load data
## Install dependencies and setup
pip install tensorflow opencv-python matplotlib

## Create dataset
Images were downloaded from Google using Chrome extension "Download All Images".
![img.png](doc/download_dataset_img.png)

## Project structure
The project consist of 6 files:
- **data_preparation.py**: contains data prepration operations
  - remove_incorrect_images():
    - Remove corrupted, mislabeled, mis-extended images, check if the file extensions are correct
  - load_data(path):
    - Load entire dataset from a path.
    - ***path***: path of the dataset (eg. 'data')
- **data_preprocessing.py**: contains operations to preprocess loaded data, before using it to training
  - split_data(data, save_model_time):
    - Split dataset to 70% of the data used for training, # 20% of the data used for validation and 10% of the data used for test the model.
    - ***data***: represents the correct data that was read and has the right format. The type of the data is tf.data.Dataset.
    - ***save_model_time***: timestamp used to distinguish saved model versions.
  - save_images(data, directory_name, save_time):
    - Save a divided data in a different directory for later usage in the predition.
    - ***data***: represents the correct data that was read and has the right format.
    - ***directory_name***: represents the name of the dataset that was generated from all the data (test, train or validation).
    - ***save_time***: timestamp used to distinguish saved model versions.
- **gpu_setup.py**: limit the gpu used percentage to avoid out of memory errors by the training using too much memory during the dataset preparation and training.
- **plotting.py**: contains some plotting function after the image dataset is loaded and a function to plot the loss and accuracy results after the training is finished.
- **model.py**: contains only model operations, like building the model based on a architecture, run trainings, validation, tests, save a pretrained model and load a pretrained model.
  - build_model(): 
    - Function used to build the model.
    - Prints model sumary after the model is created and compiled.
    - Returns the created and compiled model.
  - train_model(model, train_data, val_data): 
    - Function used to train the model using a training dataset.
    - ***model*** is the model created and compiled in _build_model()_.
    - ***train_data*** is the generated training dataset.
    - ***val_data*** is the generated validation dataset.
  - evaluate_model(model, test_data):
    - Function to evaluate performance of the model.
    - ***model*** is the model created and compiled in _build_model()_.
    - ***test_data*** is the generated test dataset.
  - test_model(model, image_path):
    - Function used to test the model using only one image.
    - ***model*** is the model created and compiled in _build_model()_.
    - ***image_path*** path of the tested image. (eg. "happy_test3.jpg", "saved_data/test1704661267.149294/happy/friends-happy-190821.jpg")
  - test_model_dataset(model, test_set):
    - Function used to test the model using a prepared test dataset.
    - ***model*** is the model created and compiled in _build_model()_.
    - ***test_set*** is the created test dataset and saved in saved_data/test{timestamp}
  - save_model(model, model_name):
    - Function used to save model into models directory
    - ***model*** is the model created and compiled in _build_model()_.
    - ***model_name*** is the name used for saving the model (eg. happysadmodel{timestamp}.h5)
  - load_model_from_path(model_name):
    - Function used to load models from models directory.
    - ***model_name*** is the name of the saved model that is desired to be loaded (eg. happysadmodel{timestamp}.h5).
- **main.py**: the main script from where all the trainings, validation, and tests are running

## Dataset
Consists of 619 files belonging to 2 classes.
The classes are:
- Sad people.
- Happy people.

All data is saved in "data" directory, that has a new folder for every class (***sad*** for images with sad people and ***happy*** for images with happy people).

## Model architecture
![img.png](doc/model_architecture.png)

## Model training
The training of the model behave as expected, with the epochs that increase and the model learn, the loss decrease and the accuracy increase, meaning that the model is learning right.

Epoch 1/20

14/14 [==============================] - ETA: 0s - loss: 0.9515 - accuracy: 0.5469

14/14 [==============================] - 17s 1s/step - loss: 0.9515 - accuracy: 0.5469 - val_loss: 0.6711 - val_accuracy: 0.5500


Epoch 2/20

 1/14 [=>............................] - ETA: 22s - loss: 0.7105 - accuracy: 0.4062

14/14 [==============================] - 12s 801ms/step - loss: 0.6566 - accuracy: 0.6049 - val_loss: 0.5788 - val_accuracy: 0.7188


Epoch 3/20

 1/14 [=>............................] - ETA: 22s - loss: 0.5767 - accuracy: 0.7188
14/14 [==============================] - 12s 810ms/step - loss: 0.5444 - accuracy: 0.7411 - val_loss: 0.5314 - val_accuracy: 0.7500


Epoch 4/20

 1/14 [=>............................] - ETA: 22s - loss: 0.4470 - accuracy: 0.7812

14/14 [==============================] - 12s 803ms/step - loss: 0.4584 - accuracy: 0.7991 - val_loss: 0.3369 - val_accuracy: 0.8562


Epoch 5/20

 1/14 [=>............................] - ETA: 21s - loss: 0.3923 - accuracy: 0.8438

14/14 [==============================] - 12s 807ms/step - loss: 0.3882 - accuracy: 0.8214 - val_loss: 0.3418 - val_accuracy: 0.8562


Epoch 6/20

 1/14 [=>............................] - ETA: 22s - loss: 0.3059 - accuracy: 0.8750

14/14 [==============================] - 12s 816ms/step - loss: 0.3794 - accuracy: 0.8371 - val_loss: 0.3592 - val_accuracy: 0.8750


Epoch 7/20

 1/14 [=>............................] - ETA: 21s - loss: 0.2775 - accuracy: 0.9062

14/14 [==============================] - 12s 802ms/step - loss: 0.3189 - accuracy: 0.8683 - val_loss: 0.3268 - val_accuracy: 0.8062


Epoch 8/20

 1/14 [=>............................] - ETA: 22s - loss: 0.0851 - accuracy: 1.0000

14/14 [==============================] - 12s 805ms/step - loss: 0.2858 - accuracy: 0.8973 - val_loss: 0.2167 - val_accuracy: 0.9438


Epoch 9/20

 1/14 [=>............................] - ETA: 22s - loss: 0.2109 - accuracy: 0.8750

14/14 [==============================] - 12s 796ms/step - loss: 0.1990 - accuracy: 0.9353 - val_loss: 0.1201 - val_accuracy: 0.9625


Epoch 10/20

 1/14 [=>............................] - ETA: 22s - loss: 0.1196 - accuracy: 1.0000

14/14 [==============================] - 12s 803ms/step - loss: 0.1343 - accuracy: 0.9621 - val_loss: 0.1046 - val_accuracy: 0.9812


Epoch 11/20

 1/14 [=>............................] - ETA: 22s - loss: 0.1359 - accuracy: 1.0000

14/14 [==============================] - 13s 859ms/step - loss: 0.1461 - accuracy: 0.9509 - val_loss: 0.1847 - val_accuracy: 0.9187


Epoch 12/20

 1/14 [=>............................] - ETA: 24s - loss: 0.0923 - accuracy: 0.9688

14/14 [==============================] - 13s 846ms/step - loss: 0.0828 - accuracy: 0.9821 - val_loss: 0.0733 - val_accuracy: 0.9812


Epoch 13/20

 1/14 [=>............................] - ETA: 21s - loss: 0.0801 - accuracy: 0.9688

14/14 [==============================] - 12s 806ms/step - loss: 0.0646 - accuracy: 0.9777 - val_loss: 0.0454 - val_accuracy: 0.9937


Epoch 14/20

 1/14 [=>............................] - ETA: 22s - loss: 0.0783 - accuracy: 1.0000

14/14 [==============================] - 12s 808ms/step - loss: 0.0656 - accuracy: 0.9821 - val_loss: 0.0373 - val_accuracy: 0.9937


Epoch 15/20

 1/14 [=>............................] - ETA: 22s - loss: 0.0239 - accuracy: 1.0000

14/14 [==============================] - 12s 803ms/step - loss: 0.0307 - accuracy: 0.9955 - val_loss: 0.0146 - val_accuracy: 1.0000


Epoch 16/20

 1/14 [=>............................] - ETA: 22s - loss: 0.0949 - accuracy: 0.9688

14/14 [==============================] - 12s 796ms/step - loss: 0.0604 - accuracy: 0.9821 - val_loss: 0.0831 - val_accuracy: 0.9625


Epoch 17/20

 1/14 [=>............................] - ETA: 21s - loss: 0.0686 - accuracy: 0.9688

14/14 [==============================] - 12s 804ms/step - loss: 0.0449 - accuracy: 0.9866 - val_loss: 0.0841 - val_accuracy: 0.9635


Epoch 18/20

 1/14 [=>............................] - ETA: 22s - loss: 0.0085 - accuracy: 1.0000

14/14 [==============================] - 12s 804ms/step - loss: 0.0143 - accuracy: 0.9978 - val_loss: 0.0804 - val_accuracy: 0.9615


Epoch 19/20

 1/14 [=>............................] - ETA: 22s - loss: 0.0070 - accuracy: 1.0000

14/14 [==============================] - 12s 816ms/step - loss: 0.0233 - accuracy: 0.9978 - val_loss: 0.0832 - val_accuracy: 0.9615


Epoch 20/20

 1/14 [=>............................] - ETA: 22s - loss: 0.0089 - accuracy: 1.0000

14/14 [==============================] - 12s 806ms/step - loss: 0.0179 - accuracy: 0.9933 - val_loss: 0.0476 - val_accuracy: 0.9866

## Model validation
Precision: 0.9166, Recall: 0.8866, Accuracy: 0.9866

## Model test
The images used for test were saved in "saved_data/test*".
Images used for testing and how the model predicted the class:

Predictions for happy test dataset:

| Image                                                                                                                                                                                                                        | Image path                                                                                   | Predicted class |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|-----------------|
| ![gdfVVm_MyCRtqpvdkt8vtSB1n_oz_CpwCq6vNMpj0S8.jpg](saved_data/test1704661267.149294/happy/gdfVVm_MyCRtqpvdkt8vtSB1n_oz_CpwCq6vNMpj0S8.jpg)                                                                                   | saved_data/test1704661267.149294/happy/gdfVVm_MyCRtqpvdkt8vtSB1n_oz_CpwCq6vNMpj0S8.jpg | happy |
| ![GettyImages-454356720.jpg](saved_data/test1704661267.149294/happy/GettyImages-454356720.jpg)                                                                                                                               | saved_data/test1704661267.149294/happy/GettyImages-454356720.jpg | happy |
| ![GettyImages-871518740.jpg](saved_data/test1704661267.149294/happy/GettyImages-871518740.jpg)                                                                                                                               | saved_data/test1704661267.149294/happy/GettyImages-871518740.jpg | happy |
| ![getty_107808336_9708069704500170_50554.jpg](saved_data/test1704661267.149294/happy/getty_107808336_9708069704500170_50554.jpg)                                                                                             | saved_data/test1704661267.149294/happy/getty_107808336_9708069704500170_50554.jpg | happy |
| ![getty_143919450_9706479704500104_51510.jpg](saved_data/test1704661267.149294/happy/getty_143919450_9706479704500104_51510.jpg)                                                                                             | saved_data/test1704661267.149294/happy/getty_143919450_9706479704500104_51510.jpg | happy |
| ![getty_152414899_97046097045006_68075.jpg](saved_data/test1704661267.149294/happy/getty_152414899_97046097045006_68075.jpg)                                                                                                 | saved_data/test1704661267.149294/happy/getty_152414899_97046097045006_68075.jpg | happy |
| ![getty_478389113_970647970450091_99776.jpg](saved_data/test1704661267.149294/happy/getty_478389113_970647970450091_99776.jpg)                                                                                               | saved_data/test1704661267.149294/happy/getty_478389113_970647970450091_99776.jpg | sad |
| ![getty_494581822_130796.jpg](saved_data/test1704661267.149294/happy/getty_494581822_130796.jpg)                                                                                                                             | saved_data/test1704661267.149294/happy/getty_494581822_130796.jpg | happy |
| ![goup-happy-people-35582464.jpg](saved_data/test1704661267.149294/happy/goup-happy-people-35582464.jpg)                                                                                                                     | saved_data/test1704661267.149294/happy/goup-happy-people-35582464.jpg | happy |
| ![goup-happy-people-group-jumping-isolated-white-background-35582232.jpg](saved_data/test1704661267.149294/happy/goup-happy-people-group-jumping-isolated-white-background-35582232.jpg)                                     | saved_data/test1704661267.149294/happy/goup-happy-people-group-jumping-isolated-white-background-35582232.jpg | happy |
| ![group-happy-people-party-isolated-white-background-31666248.jpg](saved_data/test1704661267.149294/happy/group-happy-people-party-isolated-white-background-31666248.jpg)                                                   | saved_data/test1704661267.149294/happy/group-happy-people-party-isolated-white-background-31666248.jpg | happy |
| ![group-of-happy-people-2.jpg](saved_data/test1704661267.149294/happy/group-of-happy-people-2.jpg)                                                                                                                           | saved_data/test1704661267.149294/happy/group-of-happy-people-2.jpg | happy |
| ![group-people-posing-photo-with-words-happy-bottom_577115-20873.jpg](saved_data/test1704661267.149294/happy/group-people-posing-photo-with-words-happy-bottom_577115-20873.jpg)                                             | saved_data/test1704661267.149294/happy/group-people-posing-photo-with-words-happy-bottom_577115-20873.jpg | happy |
| ![group-young-happy-people-with-their-hands-up_369728-62.jpg](saved_data/test1704661267.149294/happy/group-young-happy-people-with-their-hands-up_369728-62.jpg)                                                             | saved_data/test1704661267.149294/happy/group-young-happy-people-with-their-hands-up_369728-62.jpg | happy |
| ![habits-of-happy-people-jpg.jpg](saved_data/test1704661267.149294/happy/habits-of-happy-people-jpg.jpg)                                                                                                                     | saved_data/test1704661267.149294/happy/habits-of-happy-people-jpg.jpg | happy |
| ![hand-drawn-happy-people-jumping_23-2149092878.jpg](saved_data/test1704661267.149294/happy/hand-drawn-happy-people-jumping_23-2149092878.jpg)                                                                               | saved_data/test1704661267.149294/happy/hand-drawn-happy-people-jumping_23-2149092878.jpg | happy |
| ![Happiness-Habits-10-Things-Happy-People-Do-Before-Bed.jpg](saved_data/test1704661267.149294/happy/Happiness-Habits-10-Things-Happy-People-Do-Before-Bed.jpg)                                                               | saved_data/test1704661267.149294/happy/Happiness-Habits-10-Things-Happy-People-Do-Before-Bed.jpg | happy |
| ![Happy-Guy.jpg](saved_data/test1704661267.149294/happy/Happy-Guy.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/Happy-Guy.jpg | happy |
| ![happy-people-2.jpg](saved_data/test1704661267.149294/happy/happy-people-2.jpg)                                                                                                                                             | saved_data/test1704661267.149294/happy/happy-people-2.jpg | happy |
| ![Happy-people-800x533.jpg](saved_data/test1704661267.149294/happy/Happy-people-800x533.jpg)                                                                                                                                 | saved_data/test1704661267.149294/happy/Happy-people-800x533.jpg | happy |
| ![happy-people-crowd-board-text-8048542.jpg](saved_data/test1704661267.149294/happy/happy-people-crowd-board-text-8048542.jpg)                                                                                               | saved_data/test1704661267.149294/happy/happy-people-crowd-board-text-8048542.jpg | happy |
| ![happy-people-do-every-day.png](saved_data/test1704661267.149294/happy/happy-people-do-every-day.png)                                                                                                                       | saved_data/test1704661267.149294/happy/happy-people-do-every-day.png | happy |
| ![happy-people-group-fb.jpg](saved_data/test1704661267.149294/happy/happy-people-group-fb.jpg)                                                                                                                               | saved_data/test1704661267.149294/happy/happy-people-group-fb.jpg | happy |
| ![happy-people21.jpg](saved_data/test1704661267.149294/happy/happy-people21.jpg)                                                                                                                                             | saved_data/test1704661267.149294/happy/happy-people21.jpg | happy |
| ![happy-people3.jpg](saved_data/test1704661267.149294/happy/happy-people3.jpg)                                                                                                                                               | saved_data/test1704661267.149294/happy/happy-people3.jpg | happy |
| ![happy-woman-in-nature-at-sunset.jpg](saved_data/test1704661267.149294/happy/happy-woman-in-nature-at-sunset.jpg)                                                                                                           | saved_data/test1704661267.149294/happy/happy-woman-in-nature-at-sunset.jpg | sad |
| ![image18.jpeg](saved_data/test1704661267.149294/happy/image18.jpeg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/image18.jpeg | sad |
| ![image19.jpeg](saved_data/test1704661267.149294/happy/image19.jpeg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/image19.jpeg | happy |
| ![image2.jpeg](saved_data/test1704661267.149294/happy/image2.jpeg)                                                                                                                                                           | saved_data/test1704661267.149294/happy/image2.jpeg | sad |
| ![image20.jpeg](saved_data/test1704661267.149294/happy/image20.jpeg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/image20.jpeg | happy |
| ![image21.jpeg](saved_data/test1704661267.149294/happy/image21.jpeg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/image21.jpeg | happy |
| ![image22.jpeg](saved_data/test1704661267.149294/happy/image22.jpeg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/image22.jpeg | happy |
| ![image23.jpeg](saved_data/test1704661267.149294/happy/image23.jpeg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/image23.jpeg | happy |
| ![image24.jpeg](saved_data/test1704661267.149294/happy/image24.jpeg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/image24.jpeg | happy |
| ![image25.jpeg](saved_data/test1704661267.149294/happy/image25.jpeg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/image25.jpeg | happy |
| ![image26.jpeg](saved_data/test1704661267.149294/happy/image26.jpeg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/image26.jpeg | happy |
| ![image27.jpeg](saved_data/test1704661267.149294/happy/image27.jpeg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/image27.jpeg | happy |
| ![image28.jpeg](saved_data/test1704661267.149294/happy/image28.jpeg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/image28.jpeg | happy |
| ![image29.jpeg](saved_data/test1704661267.149294/happy/image29.jpeg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/image29.jpeg | happy |
| ![image3.jpeg](saved_data/test1704661267.149294/happy/image3.jpeg)                                                                                                                                                           | saved_data/test1704661267.149294/happy/image3.jpeg | sad |
| ![image30.jpeg](saved_data/test1704661267.149294/happy/image30.jpeg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/image30.jpeg | happy |
| ![image31.jpeg](saved_data/test1704661267.149294/happy/image31.jpeg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/image31.jpeg | happy |
| ![image32.png](saved_data/test1704661267.149294/happy/image32.png)                                                                                                                                                           | saved_data/test1704661267.149294/happy/image32.png | happy |
| ![images120.jpg](saved_data/test1704661267.149294/happy/images120.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images120.jpg | happy |
| ![images121.jpg](saved_data/test1704661267.149294/happy/images121.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images121.jpg | sad |
| ![images122.jpg](saved_data/test1704661267.149294/happy/images122.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images122.jpg | happy |
| ![images123.jpg](saved_data/test1704661267.149294/happy/images123.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images123.jpg | happy |
| ![images124.jpg](saved_data/test1704661267.149294/happy/images124.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images124.jpg | happy |
| ![images125.jpg](saved_data/test1704661267.149294/happy/images125.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images125.jpg | happy |
| ![images126.jpg](saved_data/test1704661267.149294/happy/images126.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images126.jpg | happy |
| ![images127.jpg](saved_data/test1704661267.149294/happy/images127.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images127.jpg | happy |
| ![images128.jpg](saved_data/test1704661267.149294/happy/images128.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images128.jpg | sad |
| ![images129.jpg](saved_data/test1704661267.149294/happy/images129.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images129.jpg | happy |
| ![images13.jpg](saved_data/test1704661267.149294/happy/images13.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images13.jpg | happy |
| ![images130.jpg](saved_data/test1704661267.149294/happy/images130.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images130.jpg | happy |
| ![images131.jpg](saved_data/test1704661267.149294/happy/images131.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images131.jpg | sad |
| ![images132.jpg](saved_data/test1704661267.149294/happy/images132.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images132.jpg | happy |
| ![images133.jpg](saved_data/test1704661267.149294/happy/images133.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images133.jpg | happy |
| ![images148.jpg](saved_data/test1704661267.149294/happy/images148.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images148.jpg | sad |
| ![images149.jpg](saved_data/test1704661267.149294/happy/images149.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images149.jpg | happy |
| ![images15.jpg](saved_data/test1704661267.149294/happy/images15.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images15.jpg | sad |
| ![images150.jpg](saved_data/test1704661267.149294/happy/images150.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images150.jpg | happy |
| ![images151.jpg](saved_data/test1704661267.149294/happy/images151.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images151.jpg | happy |
| ![images152.jpg](saved_data/test1704661267.149294/happy/images152.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images152.jpg | happy |
| ![images153.jpg](saved_data/test1704661267.149294/happy/images153.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images153.jpg | happy |
| ![images154.jpg](saved_data/test1704661267.149294/happy/images154.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images154.jpg | happy |
| ![images155.jpg](saved_data/test1704661267.149294/happy/images155.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images155.jpg | happy |
| ![images156.jpg](saved_data/test1704661267.149294/happy/images156.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images156.jpg | sad |
| ![images157.jpg](saved_data/test1704661267.149294/happy/images157.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images157.jpg | happy |
| ![images183.jpg](saved_data/test1704661267.149294/happy/images183.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images183.jpg | happy |
| ![images184.jpg](saved_data/test1704661267.149294/happy/images184.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images184.jpg | happy |
| ![images185.jpg](saved_data/test1704661267.149294/happy/images185.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images185.jpg | sad |
| ![images186.jpg](saved_data/test1704661267.149294/happy/images186.jpg)                                                                                                                                                       | saved_data/test1704661267.149294/happy/images186.jpg | sad |
| ![images22.jpg](saved_data/test1704661267.149294/happy/images22.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images22.jpg | happy |
| ![images23.jpg](saved_data/test1704661267.149294/happy/images23.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images23.jpg | happy |
| ![images24.jpg](saved_data/test1704661267.149294/happy/images24.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images24.jpg | sad |
| ![images31.jpg](saved_data/test1704661267.149294/happy/images31.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images31.jpg | happy |
| ![images32.jpg](saved_data/test1704661267.149294/happy/images32.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images32.jpg | sad |
| ![images33.jpg](saved_data/test1704661267.149294/happy/images33.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images33.jpg | sad |
| ![images34.jpg](saved_data/test1704661267.149294/happy/images34.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images34.jpg | happy |
| ![images35.jpg](saved_data/test1704661267.149294/happy/images35.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images35.jpg | happy |
| ![images74.jpg](saved_data/test1704661267.149294/happy/images74.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images74.jpg | happy |
| ![images75.jpg](saved_data/test1704661267.149294/happy/images75.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images75.jpg | sad |
| ![images8.jpg](saved_data/test1704661267.149294/happy/images8.jpg)                                                                                                                                                           | saved_data/test1704661267.149294/happy/images8.jpg | sad |
| ![images80.jpg](saved_data/test1704661267.149294/happy/images80.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images80.jpg | sad |
| ![images81.jpg](saved_data/test1704661267.149294/happy/images81.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images81.jpg | happy |
| ![images82.jpg](saved_data/test1704661267.149294/happy/images82.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images82.jpg | happy |
| ![images83.jpg](saved_data/test1704661267.149294/happy/images83.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images83.jpg | happy |
| ![images84.jpg](saved_data/test1704661267.149294/happy/images84.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images84.jpg | happy |
| ![images85.jpg](saved_data/test1704661267.149294/happy/images85.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images85.jpg | happy |
| ![images86.jpg](saved_data/test1704661267.149294/happy/images86.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images86.jpg | sad |
| ![images87.jpg](saved_data/test1704661267.149294/happy/images87.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images87.jpg | happy |
| ![images95.jpg](saved_data/test1704661267.149294/happy/images95.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images95.jpg | sad |
| ![images96.jpg](saved_data/test1704661267.149294/happy/images96.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images96.jpg | happy |
| ![images97.jpg](saved_data/test1704661267.149294/happy/images97.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images97.jpg | happy |
| ![images98.jpg](saved_data/test1704661267.149294/happy/images98.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images98.jpg | sad |
| ![images99.jpg](saved_data/test1704661267.149294/happy/images99.jpg)                                                                                                                                                         | saved_data/test1704661267.149294/happy/images99.jpg | happy |
| ![image_jumpstory-download20230421-155938_7a7b427.jpg](saved_data/test1704661267.149294/happy/image_jumpstory-download20230421-155938_7a7b427.jpg)                                                                           | saved_data/test1704661267.149294/happy/image_jumpstory-download20230421-155938_7a7b427.jpg | happy |
| ![ipsos-global-advisor-happiness-2022-opti.jpg](saved_data/test1704661267.149294/happy/ipsos-global-advisor-happiness-2022-opti.jpg)                                                                                         | saved_data/test1704661267.149294/happy/ipsos-global-advisor-happiness-2022-opti.jpg | sad |
| ![jumping-and-dancing-happy-people-positive-emotions-set-illustration-free-vector.jpg](saved_data/test1704661267.149294/happy/jumping-and-dancing-happy-people-positive-emotions-set-illustration-free-vector.jpg)           | saved_data/test1704661267.149294/happy/jumping-and-dancing-happy-people-positive-emotions-set-illustration-free-vector.jpg | happy |
| ![maxresdefault.jpg](saved_data/test1704661267.149294/happy/maxresdefault.jpg)                                                                                                                                               | saved_data/test1704661267.149294/happy/maxresdefault.jpg | happy |
| ![maxresdefault2.jpg](saved_data/test1704661267.149294/happy/maxresdefault2.jpg)                                                                                                                                             | saved_data/test1704661267.149294/happy/maxresdefault2.jpg | happy |
| ![MV5BMGI1NzJiMWUtMmYwMS00MDllLWJkYWYtNDVhOTY0M2U5ODdmXkEyXkFqcGdeQXVyMjA0MzYwMDY._V1_.jpg](saved_data/test1704661267.149294/happy/MV5BMGI1NzJiMWUtMmYwMS00MDllLWJkYWYtNDVhOTY0M2U5ODdmXkEyXkFqcGdeQXVyMjA0MzYwMDY._V1_.jpg) | saved_data/test1704661267.149294/happy/MV5BMGI1NzJiMWUtMmYwMS00MDllLWJkYWYtNDVhOTY0M2U5ODdmXkEyXkFqcGdeQXVyMjA0MzYwMDY._V1_.jpg | happy |
| ![o-HAPPY-facebook.jpg](saved_data/test1704661267.149294/happy/o-HAPPY-facebook.jpg)                                                                                                                                         | saved_data/test1704661267.149294/happy/o-HAPPY-facebook.jpg | sad |
| ![smile.woman_.jpg](saved_data/test1704661267.149294/happy/smile.woman_.jpg)                                                                                                                                                 | saved_data/test1704661267.149294/happy/smile.woman_.jpg | happy |
| ![Successful-year.jpg](saved_data/test1704661267.149294/happy/Successful-year.jpg)                                                                                                                                           | saved_data/test1704661267.149294/happy/Successful-year.jpg | happy |
| ![Super-Happy-People-yay.jpg](saved_data/test1704661267.149294/happy/Super-Happy-People-yay.jpg)                                                                                                                             | saved_data/test1704661267.149294/happy/Super-Happy-People-yay.jpg | sad |
| ![traitshappypeople.jpg](saved_data/test1704661267.149294/happy/traitshappypeople.jpg)                                                                                                                                       | saved_data/test1704661267.149294/happy/traitshappypeople.jpg | sad |

Number of correct predicted images 98 from a total number of 107 images.



Predictions for sad test dataset:

| Image                                                                                                                                                  | Image path                                                                                   | Predicted class |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|-----------------|
| ![12165734.jpg](saved_data/test1704661267.149294/sad/12165734.jpg) | saved_data/test1704661267.149294/sad/12165734.jpg | sad |
| ![141203-depression-stock.jpg](saved_data/test1704661267.149294/sad/141203-depression-stock.jpg) | saved_data/test1704661267.149294/sad/141203-depression-stock.jpg | sad |
| ![39843138-sad-man.jpg](saved_data/test1704661267.149294/sad/39843138-sad-man.jpg) | saved_data/test1704661267.149294/sad/39843138-sad-man.jpg | sad |
| ![7RNXwSxCAKL8vGtXG2ZkyD-1200-80.jpg](saved_data/test1704661267.149294/sad/7RNXwSxCAKL8vGtXG2ZkyD-1200-80.jpg) | saved_data/test1704661267.149294/sad/7RNXwSxCAKL8vGtXG2ZkyD-1200-80.jpg | sad |
| ![8iAb9k4aT.jpg](saved_data/test1704661267.149294/sad/8iAb9k4aT.jpg) | saved_data/test1704661267.149294/sad/8iAb9k4aT.jpg | sad |
| ![960x0.jpg](saved_data/test1704661267.149294/sad/960x0.jpg) | saved_data/test1704661267.149294/sad/960x0.jpg | sad |
| ![ANLem4bnFq0ysIBZ60imcOQ2_mzu7Z802BGAclqe63nws64-c-mo.jpg](saved_data/test1704661267.149294/sad/ANLem4bnFq0ysIBZ60imcOQ2_mzu7Z802BGAclqe63nws64-c-mo.jpg) | saved_data/test1704661267.149294/sad/ANLem4bnFq0ysIBZ60imcOQ2_mzu7Z802BGAclqe63nws64-c-mo.jpg | happy |
| ![hqdefault.jpg](saved_data/test1704661267.149294/sad/hqdefault.jpg) | saved_data/test1704661267.149294/sad/hqdefault.jpg | sad |
| ![image-20160914-4963-19knfh1.jpg](saved_data/test1704661267.149294/sad/image-20160914-4963-19knfh1.jpg) | saved_data/test1704661267.149294/sad/image-20160914-4963-19knfh1.jpg | sad |
| ![image-asset.jpeg](saved_data/test1704661267.149294/sad/image-asset.jpeg) | saved_data/test1704661267.149294/sad/image-asset.jpeg | sad |
| ![image.jpeg](saved_data/test1704661267.149294/sad/image.jpeg) | saved_data/test1704661267.149294/sad/image.jpeg | sad |
| ![image10.jpeg](saved_data/test1704661267.149294/sad/image10.jpeg) | saved_data/test1704661267.149294/sad/image10.jpeg | sad |
| ![image19.jpeg](saved_data/test1704661267.149294/sad/image19.jpeg) | saved_data/test1704661267.149294/sad/image19.jpeg | sad |
| ![image2.jpeg](saved_data/test1704661267.149294/sad/image2.jpeg) | saved_data/test1704661267.149294/sad/image2.jpeg | sad |
| ![image20.jpeg](saved_data/test1704661267.149294/sad/image20.jpeg) | saved_data/test1704661267.149294/sad/image20.jpeg | sad |
| ![image21.jpeg](saved_data/test1704661267.149294/sad/image21.jpeg) | saved_data/test1704661267.149294/sad/image21.jpeg | sad |
| ![image22.jpeg](saved_data/test1704661267.149294/sad/image22.jpeg) | saved_data/test1704661267.149294/sad/image22.jpeg | sad |
| ![image23.jpeg](saved_data/test1704661267.149294/sad/image23.jpeg) | saved_data/test1704661267.149294/sad/image23.jpeg | sad |
| ![image24.jpeg](saved_data/test1704661267.149294/sad/image24.jpeg) | saved_data/test1704661267.149294/sad/image24.jpeg | sad |
| ![image25.jpeg](saved_data/test1704661267.149294/sad/image25.jpeg) | saved_data/test1704661267.149294/sad/image25.jpeg | sad |
| ![image26.jpeg](saved_data/test1704661267.149294/sad/image26.jpeg) | saved_data/test1704661267.149294/sad/image26.jpeg | sad |
| ![image27.jpeg](saved_data/test1704661267.149294/sad/image27.jpeg) | saved_data/test1704661267.149294/sad/image27.jpeg | sad |
| ![image28.jpeg](saved_data/test1704661267.149294/sad/image28.jpeg) | saved_data/test1704661267.149294/sad/image28.jpeg | sad |
| ![image29.jpeg](saved_data/test1704661267.149294/sad/image29.jpeg) | saved_data/test1704661267.149294/sad/image29.jpeg | sad |
| ![image3.jpeg](saved_data/test1704661267.149294/sad/image3.jpeg) | saved_data/test1704661267.149294/sad/image3.jpeg | sad |
| ![image30.jpeg](saved_data/test1704661267.149294/sad/image30.jpeg) | saved_data/test1704661267.149294/sad/image30.jpeg | sad |
| ![image31.jpeg](saved_data/test1704661267.149294/sad/image31.jpeg) | saved_data/test1704661267.149294/sad/image31.jpeg | sad |
| ![image32.png](saved_data/test1704661267.149294/sad/image32.png) | saved_data/test1704661267.149294/sad/image32.png | sad |
| ![image4.jpeg](saved_data/test1704661267.149294/sad/image4.jpeg) | saved_data/test1704661267.149294/sad/image4.jpeg | sad |
| ![image5.jpeg](saved_data/test1704661267.149294/sad/image5.jpeg) | saved_data/test1704661267.149294/sad/image5.jpeg | happy |
| ![images108.jpg](saved_data/test1704661267.149294/sad/images108.jpg) | saved_data/test1704661267.149294/sad/images108.jpg | sad |
| ![images109.jpg](saved_data/test1704661267.149294/sad/images109.jpg) | saved_data/test1704661267.149294/sad/images109.jpg | sad |
| ![images11.jpg](saved_data/test1704661267.149294/sad/images11.jpg) | saved_data/test1704661267.149294/sad/images11.jpg | sad |
| ![images110.jpg](saved_data/test1704661267.149294/sad/images110.jpg) | saved_data/test1704661267.149294/sad/images110.jpg | sad |
| ![images111.jpg](saved_data/test1704661267.149294/sad/images111.jpg) | saved_data/test1704661267.149294/sad/images111.jpg | sad |
| ![images112.jpg](saved_data/test1704661267.149294/sad/images112.jpg) | saved_data/test1704661267.149294/sad/images112.jpg | sad |
| ![images113.jpg](saved_data/test1704661267.149294/sad/images113.jpg) | saved_data/test1704661267.149294/sad/images113.jpg | sad |
| ![images114.jpg](saved_data/test1704661267.149294/sad/images114.jpg) | saved_data/test1704661267.149294/sad/images114.jpg | sad |
| ![images115.jpg](saved_data/test1704661267.149294/sad/images115.jpg) | saved_data/test1704661267.149294/sad/images115.jpg | sad |
| ![images116.jpg](saved_data/test1704661267.149294/sad/images116.jpg) | saved_data/test1704661267.149294/sad/images116.jpg | sad |
| ![images117.jpg](saved_data/test1704661267.149294/sad/images117.jpg) | saved_data/test1704661267.149294/sad/images117.jpg | sad |
| ![images118.jpg](saved_data/test1704661267.149294/sad/images118.jpg) | saved_data/test1704661267.149294/sad/images118.jpg | sad |
| ![images119.jpg](saved_data/test1704661267.149294/sad/images119.jpg) | saved_data/test1704661267.149294/sad/images119.jpg | sad |
| ![images12.jpg](saved_data/test1704661267.149294/sad/images12.jpg) | saved_data/test1704661267.149294/sad/images12.jpg | sad |
| ![images120.jpg](saved_data/test1704661267.149294/sad/images120.jpg) | saved_data/test1704661267.149294/sad/images120.jpg | sad |
| ![images121.jpg](saved_data/test1704661267.149294/sad/images121.jpg) | saved_data/test1704661267.149294/sad/images121.jpg | sad |
| ![images122.jpg](saved_data/test1704661267.149294/sad/images122.jpg) | saved_data/test1704661267.149294/sad/images122.jpg | sad |
| ![images123.jpg](saved_data/test1704661267.149294/sad/images123.jpg) | saved_data/test1704661267.149294/sad/images123.jpg | sad |
| ![images124.jpg](saved_data/test1704661267.149294/sad/images124.jpg) | saved_data/test1704661267.149294/sad/images124.jpg | sad |
| ![images125.jpg](saved_data/test1704661267.149294/sad/images125.jpg) | saved_data/test1704661267.149294/sad/images125.jpg | sad |
| ![images126.jpg](saved_data/test1704661267.149294/sad/images126.jpg) | saved_data/test1704661267.149294/sad/images126.jpg | sad |
| ![images127.jpg](saved_data/test1704661267.149294/sad/images127.jpg) | saved_data/test1704661267.149294/sad/images127.jpg | sad |
| ![images128.jpg](saved_data/test1704661267.149294/sad/images128.jpg) | saved_data/test1704661267.149294/sad/images128.jpg | sad |
| ![images129.jpg](saved_data/test1704661267.149294/sad/images129.jpg) | saved_data/test1704661267.149294/sad/images129.jpg | sad |
| ![images13.jpg](saved_data/test1704661267.149294/sad/images13.jpg) | saved_data/test1704661267.149294/sad/images13.jpg | sad |
| ![images130.jpg](saved_data/test1704661267.149294/sad/images130.jpg) | saved_data/test1704661267.149294/sad/images130.jpg | sad |
| ![images131.jpg](saved_data/test1704661267.149294/sad/images131.jpg) | saved_data/test1704661267.149294/sad/images131.jpg | sad |
| ![images132.jpg](saved_data/test1704661267.149294/sad/images132.jpg) | saved_data/test1704661267.149294/sad/images132.jpg | sad |
| ![images133.jpg](saved_data/test1704661267.149294/sad/images133.jpg) | saved_data/test1704661267.149294/sad/images133.jpg | sad |
| ![images134.jpg](saved_data/test1704661267.149294/sad/images134.jpg) | saved_data/test1704661267.149294/sad/images134.jpg | sad |
| ![images135.jpg](saved_data/test1704661267.149294/sad/images135.jpg) | saved_data/test1704661267.149294/sad/images135.jpg | sad |
| ![images136.jpg](saved_data/test1704661267.149294/sad/images136.jpg) | saved_data/test1704661267.149294/sad/images136.jpg | sad |
| ![images137.jpg](saved_data/test1704661267.149294/sad/images137.jpg) | saved_data/test1704661267.149294/sad/images137.jpg | sad |
| ![images138.jpg](saved_data/test1704661267.149294/sad/images138.jpg) | saved_data/test1704661267.149294/sad/images138.jpg | sad |
| ![images139.jpg](saved_data/test1704661267.149294/sad/images139.jpg) | saved_data/test1704661267.149294/sad/images139.jpg | sad |
| ![images14.jpg](saved_data/test1704661267.149294/sad/images14.jpg) | saved_data/test1704661267.149294/sad/images14.jpg | happy |
| ![images140.jpg](saved_data/test1704661267.149294/sad/images140.jpg) | saved_data/test1704661267.149294/sad/images140.jpg | sad |
| ![images141.jpg](saved_data/test1704661267.149294/sad/images141.jpg) | saved_data/test1704661267.149294/sad/images141.jpg | sad |
| ![images142.jpg](saved_data/test1704661267.149294/sad/images142.jpg) | saved_data/test1704661267.149294/sad/images142.jpg | sad |
| ![images143.jpg](saved_data/test1704661267.149294/sad/images143.jpg) | saved_data/test1704661267.149294/sad/images143.jpg | sad |
| ![images144.jpg](saved_data/test1704661267.149294/sad/images144.jpg) | saved_data/test1704661267.149294/sad/images144.jpg | sad |
| ![images145.jpg](saved_data/test1704661267.149294/sad/images145.jpg) | saved_data/test1704661267.149294/sad/images145.jpg | sad |
| ![images146.jpg](saved_data/test1704661267.149294/sad/images146.jpg) | saved_data/test1704661267.149294/sad/images146.jpg | sad |
| ![images147.jpg](saved_data/test1704661267.149294/sad/images147.jpg) | saved_data/test1704661267.149294/sad/images147.jpg | sad |
| ![images148.jpg](saved_data/test1704661267.149294/sad/images148.jpg) | saved_data/test1704661267.149294/sad/images148.jpg | sad |
| ![images149.jpg](saved_data/test1704661267.149294/sad/images149.jpg) | saved_data/test1704661267.149294/sad/images149.jpg | sad |
| ![images15.jpg](saved_data/test1704661267.149294/sad/images15.jpg) | saved_data/test1704661267.149294/sad/images15.jpg | sad |
| ![images150.jpg](saved_data/test1704661267.149294/sad/images150.jpg) | saved_data/test1704661267.149294/sad/images150.jpg | sad |
| ![images151.jpg](saved_data/test1704661267.149294/sad/images151.jpg) | saved_data/test1704661267.149294/sad/images151.jpg | sad |
| ![images152.jpg](saved_data/test1704661267.149294/sad/images152.jpg) | saved_data/test1704661267.149294/sad/images152.jpg | sad |
| ![images153.jpg](saved_data/test1704661267.149294/sad/images153.jpg) | saved_data/test1704661267.149294/sad/images153.jpg | sad |
| ![images154.jpg](saved_data/test1704661267.149294/sad/images154.jpg) | saved_data/test1704661267.149294/sad/images154.jpg | sad |
| ![images155.jpg](saved_data/test1704661267.149294/sad/images155.jpg) | saved_data/test1704661267.149294/sad/images155.jpg | sad |
| ![images156.jpg](saved_data/test1704661267.149294/sad/images156.jpg) | saved_data/test1704661267.149294/sad/images156.jpg | sad |
| ![images157.jpg](saved_data/test1704661267.149294/sad/images157.jpg) | saved_data/test1704661267.149294/sad/images157.jpg | sad |
| ![images158.jpg](saved_data/test1704661267.149294/sad/images158.jpg) | saved_data/test1704661267.149294/sad/images158.jpg | sad |
| ![images159.jpg](saved_data/test1704661267.149294/sad/images159.jpg) | saved_data/test1704661267.149294/sad/images159.jpg | sad |
| ![images183.jpg](saved_data/test1704661267.149294/sad/images183.jpg) | saved_data/test1704661267.149294/sad/images183.jpg | sad |
| ![images184.jpg](saved_data/test1704661267.149294/sad/images184.jpg) | saved_data/test1704661267.149294/sad/images184.jpg | happy |
| ![images185.jpg](saved_data/test1704661267.149294/sad/images185.jpg) | saved_data/test1704661267.149294/sad/images185.jpg | sad |
| ![images186.jpg](saved_data/test1704661267.149294/sad/images186.jpg) | saved_data/test1704661267.149294/sad/images186.jpg | sad |
| ![images187.jpg](saved_data/test1704661267.149294/sad/images187.jpg) | saved_data/test1704661267.149294/sad/images187.jpg | sad |
| ![images28.jpg](saved_data/test1704661267.149294/sad/images28.jpg) | saved_data/test1704661267.149294/sad/images28.jpg | sad |
| ![images29.jpg](saved_data/test1704661267.149294/sad/images29.jpg) | saved_data/test1704661267.149294/sad/images29.jpg | sad |
| ![images48.jpg](saved_data/test1704661267.149294/sad/images48.jpg) | saved_data/test1704661267.149294/sad/images48.jpg | sad |
| ![images49.jpg](saved_data/test1704661267.149294/sad/images49.jpg) | saved_data/test1704661267.149294/sad/images49.jpg | sad |
| ![images5.jpg](saved_data/test1704661267.149294/sad/images5.jpg) | saved_data/test1704661267.149294/sad/images5.jpg | sad |
| ![images50.jpg](saved_data/test1704661267.149294/sad/images50.jpg) | saved_data/test1704661267.149294/sad/images50.jpg | sad |
| ![images51.jpg](saved_data/test1704661267.149294/sad/images51.jpg) | saved_data/test1704661267.149294/sad/images51.jpg | happy |
| ![images75.jpg](saved_data/test1704661267.149294/sad/images75.jpg) | saved_data/test1704661267.149294/sad/images75.jpg | happy |
| ![images76.jpg](saved_data/test1704661267.149294/sad/images76.jpg) | saved_data/test1704661267.149294/sad/images76.jpg | sad |
| ![images8.jpg](saved_data/test1704661267.149294/sad/images8.jpg) | saved_data/test1704661267.149294/sad/images8.jpg | sad |
| ![images80.jpg](saved_data/test1704661267.149294/sad/images80.jpg) | saved_data/test1704661267.149294/sad/images80.jpg | sad |
| ![images81.jpg](saved_data/test1704661267.149294/sad/images81.jpg) | saved_data/test1704661267.149294/sad/images81.jpg | happy |
| ![images82.jpg](saved_data/test1704661267.149294/sad/images82.jpg) | saved_data/test1704661267.149294/sad/images82.jpg | sad |
| ![images83.jpg](saved_data/test1704661267.149294/sad/images83.jpg) | saved_data/test1704661267.149294/sad/images83.jpg | sad |
| ![images89.jpg](saved_data/test1704661267.149294/sad/images89.jpg) | saved_data/test1704661267.149294/sad/images89.jpg | sad |
| ![images9.jpg](saved_data/test1704661267.149294/sad/images9.jpg) | saved_data/test1704661267.149294/sad/images9.jpg | sad |
| ![images90.jpg](saved_data/test1704661267.149294/sad/images90.jpg) | saved_data/test1704661267.149294/sad/images90.jpg | happy |
| ![images91.jpg](saved_data/test1704661267.149294/sad/images91.jpg) | saved_data/test1704661267.149294/sad/images91.jpg | sad |

Number of correct sad people predictions 104 from a total number of 112 images.
