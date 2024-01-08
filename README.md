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

14/14 [==============================] - 12s 804ms/step - loss: 0.0449 - accuracy: 0.9866 - val_loss: 0.0236 - val_accuracy: 1.0000


Epoch 18/20

 1/14 [=>............................] - ETA: 22s - loss: 0.0085 - accuracy: 1.0000

14/14 [==============================] - 12s 804ms/step - loss: 0.0143 - accuracy: 0.9978 - val_loss: 0.0120 - val_accuracy: 1.0000


Epoch 19/20

 1/14 [=>............................] - ETA: 22s - loss: 0.0070 - accuracy: 1.0000

14/14 [==============================] - 12s 816ms/step - loss: 0.0233 - accuracy: 0.9978 - val_loss: 0.0135 - val_accuracy: 1.0000


Epoch 20/20

 1/14 [=>............................] - ETA: 22s - loss: 0.0089 - accuracy: 1.0000

14/14 [==============================] - 12s 806ms/step - loss: 0.0179 - accuracy: 0.9933 - val_loss: 0.0076 - val_accuracy: 1.0000

## Model validation
Precision: 1.0, Recall: 1.0, Accuracy: 1.0

## Model test
The images used for test were saved in "saved_data/test*".
Images used for testing and how the model predicted the class:

Predictions for happy test dataset:

| Image                                                                                                                                                                                                                        | Image path                                                                                   | Predicted class |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|-----------------|
| ![05-12-21-happy-people.jpg](./saved_data/test1704661267.149294/happy/05-12-21-happy-people.jpg)                                                                                                                             | saved_data\test1704661267.149294\happy\05-12-21-happy-people.jpg                             | happy           |
| ![1000_F_165246984_Ihe5LVattiq8zEPivcPqrtM85x7noWJw.jpg](saved_data\test1704661267.149294\happy\1000_F_165246984_Ihe5LVattiq8zEPivcPqrtM85x7noWJw.jpg)                                                                       | saved_data\test1704661267.149294\happy\1000_F_165246984_Ihe5LVattiq8zEPivcPqrtM85x7noWJw.jpg | happy           |
| ![1000_F_42220133_toAU6USGY9jVr2XJFLssfG00cSYIJ173.jpg](saved_data\test1704661267.149294\happy\1000_F_42220133_toAU6USGY9jVr2XJFLssfG00cSYIJ173.jpg)                                                                         | saved_data\test1704661267.149294\happy\1000_F_42220133_toAU6USGY9jVr2XJFLssfG00cSYIJ173.jpg | happy |
| ![170404-happy-workers-feature.jpg](saved_data\test1704661267.149294\happy\170404-happy-workers-feature.jpg)                                                                                                                 | saved_data\test1704661267.149294\happy\170404-happy-workers-feature.jpg | happy |
| ![1920px-face-smile.svg_.png](saved_data\test1704661267.149294\happy\1920px-face-smile.svg_.png)                                                                                                                             | saved_data\test1704661267.149294\happy\1920px-face-smile.svg_.png | happy |
| ![1961996_stock-photo-group-of-happy-people.jpg](saved_data\test1704661267.149294\happy\1961996_stock-photo-group-of-happy-people.jpg)                                                                                       | saved_data\test1704661267.149294\happy\1961996_stock-photo-group-of-happy-people.jpg | happy |
| ![1HEoLBLidT2u4mhJ0oiDgig.png](saved_data\test1704661267.149294\happy\1HEoLBLidT2u4mhJ0oiDgig.png)                                                                                                                           | saved_data\test1704661267.149294\happy\1HEoLBLidT2u4mhJ0oiDgig.png | happy |
| ![1zgJ8mDXVYwNY_5KkZr9Wzw.jpeg](saved_data\test1704661267.149294\happy\1zgJ8mDXVYwNY_5KkZr9Wzw.jpeg)                                                                                                                         | saved_data\test1704661267.149294\happy\1zgJ8mDXVYwNY_5KkZr9Wzw.jpeg | happy |
| ![20150413185238-secrets-happy-entrepreneurs-woman-gratitude-rainbow-.jpeg](saved_data\test1704661267.149294\happy\20150413185238-secrets-happy-entrepreneurs-woman-gratitude-rainbow-.jpeg)                                 | saved_data\test1704661267.149294\happy\20150413185238-secrets-happy-entrepreneurs-woman-gratitude-rainbow-.jpeg | sad |
| ![2983960_stock-photo-happy-people.jpg](saved_data\test1704661267.149294\happy\2983960_stock-photo-happy-people.jpg)                                                                                                         | saved_data\test1704661267.149294\happy\2983960_stock-photo-happy-people.jpg | happy |
| ![35438_hd.jpg](saved_data\test1704661267.149294\happy\35438_hd.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\35438_hd.jpg | happy |
| ![360_F_484873483_hg1ofIdXbMha5lKEDG3hJBrwKh1oikTq.jpg](saved_data\test1704661267.149294\happy\360_F_484873483_hg1ofIdXbMha5lKEDG3hJBrwKh1oikTq.jpg)                                                                         | saved_data\test1704661267.149294\happy\360_F_484873483_hg1ofIdXbMha5lKEDG3hJBrwKh1oikTq.jpg | happy |
| ![56f455011e0000b300705475.jpeg](saved_data\test1704661267.149294\happy\56f455011e0000b300705475.jpeg)                                                                                                                       | saved_data\test1704661267.149294\happy\56f455011e0000b300705475.jpeg | happy |
| ![7-principles-of-successful-and-happy-people.png](saved_data\test1704661267.149294\happy\7-principles-of-successful-and-happy-people.png)                                                                                   | saved_data\test1704661267.149294\happy\7-principles-of-successful-and-happy-people.png | happy |
| ![89ca5d41335b4f9207b9cf03538a7dbd63497e474912837562cb9f58809ac32f.png](saved_data\test1704661267.149294\happy\89ca5d41335b4f9207b9cf03538a7dbd63497e474912837562cb9f58809ac32f.png)                                         | saved_data\test1704661267.149294\happy\89ca5d41335b4f9207b9cf03538a7dbd63497e474912837562cb9f58809ac32f.png | happy |
| ![8e06de1bf2171da2312b6de61c61e4bc.jpg](saved_data\test1704661267.149294\happy\8e06de1bf2171da2312b6de61c61e4bc.jpg)                                                                                                         | saved_data\test1704661267.149294\happy\8e06de1bf2171da2312b6de61c61e4bc.jpg | happy |
| ![960x0-1.jpg](saved_data\test1704661267.149294\happy\960x0-1.jpg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\960x0-1.jpg | happy |
| ![ANLem4bnFq0ysIBZ60imcOQ2_mzu7Z802BGAclqe63nws64-c-mo.jpg](saved_data\test1704661267.149294\happy\ANLem4bnFq0ysIBZ60imcOQ2_mzu7Z802BGAclqe63nws64-c-mo.jpg)                                                                 | saved_data\test1704661267.149294\happy\ANLem4bnFq0ysIBZ60imcOQ2_mzu7Z802BGAclqe63nws64-c-mo.jpg | happy |
| ![A_Sep20_14_1189155141.jpg](saved_data\test1704661267.149294\happy\A_Sep20_14_1189155141.jpg)                                                                                                                               | saved_data\test1704661267.149294\happy\A_Sep20_14_1189155141.jpg | happy |
| ![BoostYourBrainmain_0.jpg](saved_data\test1704661267.149294\happy\BoostYourBrainmain_0.jpg)                                                                                                                                 | saved_data\test1704661267.149294\happy\BoostYourBrainmain_0.jpg | happy |
| ![business-people-succesful-celebrating-group-successful-39416686-800x500.jpg](saved_data\test1704661267.149294\happy\business-people-succesful-celebrating-group-successful-39416686-800x500.jpg)                           | saved_data\test1704661267.149294\happy\business-people-succesful-celebrating-group-successful-39416686-800x500.jpg | happy |
| ![compassion-900x387.jpg](saved_data\test1704661267.149294\happy\compassion-900x387.jpg)                                                                                                                                     | saved_data\test1704661267.149294\happy\compassion-900x387.jpg | happy |
| ![diverse-friends-students-shoot_53876-47012.jpg](saved_data\test1704661267.149294\happy\diverse-friends-students-shoot_53876-47012.jpg)                                                                                     | saved_data\test1704661267.149294\happy\diverse-friends-students-shoot_53876-47012.jpg | happy |
| ![Duggar-Family-Secrets-Are-Exposed-in-New-Docuseries-Featuring-Jill-and-Amy-featured.png](saved_data\test1704661267.149294\happy\Duggar-Family-Secrets-Are-Exposed-in-New-Docuseries-Featuring-Jill-and-Amy-featured.png)   | saved_data\test1704661267.149294\happy\Duggar-Family-Secrets-Are-Exposed-in-New-Docuseries-Featuring-Jill-and-Amy-featured.png | happy |
| ![es_27x40_pre_final_en-us_cps_custom-c5a0cc5b4b5b0d8a651ee346a042970c45cf3475.jpg](saved_data\test1704661267.149294\happy\es_27x40_pre_final_en-us_cps_custom-c5a0cc5b4b5b0d8a651ee346a042970c45cf3475.jpg)                 | saved_data\test1704661267.149294\happy\es_27x40_pre_final_en-us_cps_custom-c5a0cc5b4b5b0d8a651ee346a042970c45cf3475.jpg | happy |
| ![file-20230208-27-3jttof.jpg](saved_data\test1704661267.149294\happy\file-20230208-27-3jttof.jpg)                                                                                                                           | saved_data\test1704661267.149294\happy\file-20230208-27-3jttof.jpg | happy |
| ![friends-happy-190821.jpg](saved_data\test1704661267.149294\happy\friends-happy-190821.jpg)                                                                                                                                 | saved_data\test1704661267.149294\happy\friends-happy-190821.jpg | happy |
| ![friends_190412.jpg](saved_data\test1704661267.149294\happy\friends_190412.jpg)                                                                                                                                             | saved_data\test1704661267.149294\happy\friends_190412.jpg | happy |
| ![gdfVVm_MyCRtqpvdkt8vtSB1n_oz_CpwCq6vNMpj0S8.jpg](saved_data\test1704661267.149294\happy\gdfVVm_MyCRtqpvdkt8vtSB1n_oz_CpwCq6vNMpj0S8.jpg)                                                                                   | saved_data\test1704661267.149294\happy\gdfVVm_MyCRtqpvdkt8vtSB1n_oz_CpwCq6vNMpj0S8.jpg | happy |
| ![GettyImages-454356720.jpg](saved_data\test1704661267.149294\happy\GettyImages-454356720.jpg)                                                                                                                               | saved_data\test1704661267.149294\happy\GettyImages-454356720.jpg | happy |
| ![GettyImages-871518740.jpg](saved_data\test1704661267.149294\happy\GettyImages-871518740.jpg)                                                                                                                               | saved_data\test1704661267.149294\happy\GettyImages-871518740.jpg | happy |
| ![getty_107808336_9708069704500170_50554.jpg](saved_data\test1704661267.149294\happy\getty_107808336_9708069704500170_50554.jpg)                                                                                             | saved_data\test1704661267.149294\happy\getty_107808336_9708069704500170_50554.jpg | happy |
| ![getty_143919450_9706479704500104_51510.jpg](saved_data\test1704661267.149294\happy\getty_143919450_9706479704500104_51510.jpg)                                                                                             | saved_data\test1704661267.149294\happy\getty_143919450_9706479704500104_51510.jpg | happy |
| ![getty_152414899_97046097045006_68075.jpg](saved_data\test1704661267.149294\happy\getty_152414899_97046097045006_68075.jpg)                                                                                                 | saved_data\test1704661267.149294\happy\getty_152414899_97046097045006_68075.jpg | happy |
| ![getty_478389113_970647970450091_99776.jpg](saved_data\test1704661267.149294\happy\getty_478389113_970647970450091_99776.jpg)                                                                                               | saved_data\test1704661267.149294\happy\getty_478389113_970647970450091_99776.jpg | sad |
| ![getty_494581822_130796.jpg](saved_data\test1704661267.149294\happy\getty_494581822_130796.jpg)                                                                                                                             | saved_data\test1704661267.149294\happy\getty_494581822_130796.jpg | happy |
| ![goup-happy-people-35582464.jpg](saved_data\test1704661267.149294\happy\goup-happy-people-35582464.jpg)                                                                                                                     | saved_data\test1704661267.149294\happy\goup-happy-people-35582464.jpg | happy |
| ![goup-happy-people-group-jumping-isolated-white-background-35582232.jpg](saved_data\test1704661267.149294\happy\goup-happy-people-group-jumping-isolated-white-background-35582232.jpg)                                     | saved_data\test1704661267.149294\happy\goup-happy-people-group-jumping-isolated-white-background-35582232.jpg | happy |
| ![group-happy-people-party-isolated-white-background-31666248.jpg](saved_data\test1704661267.149294\happy\group-happy-people-party-isolated-white-background-31666248.jpg)                                                   | saved_data\test1704661267.149294\happy\group-happy-people-party-isolated-white-background-31666248.jpg | happy |
| ![group-of-happy-people-2.jpg](saved_data\test1704661267.149294\happy\group-of-happy-people-2.jpg)                                                                                                                           | saved_data\test1704661267.149294\happy\group-of-happy-people-2.jpg | happy |
| ![group-people-posing-photo-with-words-happy-bottom_577115-20873.jpg](saved_data\test1704661267.149294\happy\group-people-posing-photo-with-words-happy-bottom_577115-20873.jpg)                                             | saved_data\test1704661267.149294\happy\group-people-posing-photo-with-words-happy-bottom_577115-20873.jpg | happy |
| ![group-young-happy-people-with-their-hands-up_369728-62.jpg](saved_data\test1704661267.149294\happy\group-young-happy-people-with-their-hands-up_369728-62.jpg)                                                             | saved_data\test1704661267.149294\happy\group-young-happy-people-with-their-hands-up_369728-62.jpg | happy |
| ![habits-of-happy-people-jpg.jpg](saved_data\test1704661267.149294\happy\habits-of-happy-people-jpg.jpg)                                                                                                                     | saved_data\test1704661267.149294\happy\habits-of-happy-people-jpg.jpg | happy |
| ![hand-drawn-happy-people-jumping_23-2149092878.jpg](saved_data\test1704661267.149294\happy\hand-drawn-happy-people-jumping_23-2149092878.jpg)                                                                               | saved_data\test1704661267.149294\happy\hand-drawn-happy-people-jumping_23-2149092878.jpg | happy |
| ![Happiness-Habits-10-Things-Happy-People-Do-Before-Bed.jpg](saved_data\test1704661267.149294\happy\Happiness-Habits-10-Things-Happy-People-Do-Before-Bed.jpg)                                                               | saved_data\test1704661267.149294\happy\Happiness-Habits-10-Things-Happy-People-Do-Before-Bed.jpg | happy |
| ![Happy-Guy.jpg](saved_data\test1704661267.149294\happy\Happy-Guy.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\Happy-Guy.jpg | happy |
| ![happy-people-2.jpg](saved_data\test1704661267.149294\happy\happy-people-2.jpg)                                                                                                                                             | saved_data\test1704661267.149294\happy\happy-people-2.jpg | happy |
| ![Happy-people-800x533.jpg](saved_data\test1704661267.149294\happy\Happy-people-800x533.jpg)                                                                                                                                 | saved_data\test1704661267.149294\happy\Happy-people-800x533.jpg | happy |
| ![happy-people-crowd-board-text-8048542.jpg](saved_data\test1704661267.149294\happy\happy-people-crowd-board-text-8048542.jpg)                                                                                               | saved_data\test1704661267.149294\happy\happy-people-crowd-board-text-8048542.jpg | happy |
| ![happy-people-do-every-day.png](saved_data\test1704661267.149294\happy\happy-people-do-every-day.png)                                                                                                                       | saved_data\test1704661267.149294\happy\happy-people-do-every-day.png | happy |
| ![happy-people-group-fb.jpg](saved_data\test1704661267.149294\happy\happy-people-group-fb.jpg)                                                                                                                               | saved_data\test1704661267.149294\happy\happy-people-group-fb.jpg | happy |
| ![happy-people.jpeg](saved_data\test1704661267.149294\happy\happy-people.jpeg)                                                                                                                                               | saved_data\test1704661267.149294\happy\happy-people.jpeg | happy |
| ![Happy-People.jpg](saved_data\test1704661267.149294\happy\Happy-People.jpg)                                                                                                                                                 | saved_data\test1704661267.149294\happy\Happy-People.jpg | happy |
| ![happy-people12.jpg](saved_data\test1704661267.149294\happy\happy-people12.jpg)                                                                                                                                             | saved_data\test1704661267.149294\happy\happy-people12.jpg | happy |
| ![happy-people2.jpg](saved_data\test1704661267.149294\happy\happy-people2.jpg)                                                                                                                                               | saved_data\test1704661267.149294\happy\happy-people2.jpg | happy |
| ![happy-people21.jpg](saved_data\test1704661267.149294\happy\happy-people21.jpg)                                                                                                                                             | saved_data\test1704661267.149294\happy\happy-people21.jpg | happy |
| ![happy-people3.jpg](saved_data\test1704661267.149294\happy\happy-people3.jpg)                                                                                                                                               | saved_data\test1704661267.149294\happy\happy-people3.jpg | happy |
| ![happy-woman-in-nature-at-sunset.jpg](saved_data\test1704661267.149294\happy\happy-woman-in-nature-at-sunset.jpg)                                                                                                           | saved_data\test1704661267.149294\happy\happy-woman-in-nature-at-sunset.jpg | sad |
| ![Happy.jpg](saved_data\test1704661267.149294\happy\Happy.jpg)                                                                                                                                                               | saved_data\test1704661267.149294\happy\Happy.jpg | happy |
| ![happypeople-1024x679.jpg](saved_data\test1704661267.149294\happy\happypeople-1024x679.jpg)                                                                                                                                 | saved_data\test1704661267.149294\happy\happypeople-1024x679.jpg | happy |
| ![happy_1_1678616873966_1678616915228_1678616915228.jpg](saved_data\test1704661267.149294\happy\happy_1_1678616873966_1678616915228_1678616915228.jpg)                                                                       | saved_data\test1704661267.149294\happy\happy_1_1678616873966_1678616915228_1678616915228.jpg | happy |
| ![how-happy-are-healthy-people.jpg](saved_data\test1704661267.149294\happy\how-happy-are-healthy-people.jpg)                                                                                                                 | saved_data\test1704661267.149294\happy\how-happy-are-healthy-people.jpg | happy |
| ![How_Happy_Are_People_at_Work.jpg](saved_data\test1704661267.149294\happy\How_Happy_Are_People_at_Work.jpg)                                                                                                                 | saved_data\test1704661267.149294\happy\How_Happy_Are_People_at_Work.jpg | happy |
| ![i00YzYyLTkxY2ItY2I3OWE3NDBmNDVmXkEyXkFqcGdeQXVyMjkwOTAyMDU._V1_FMjpg_UX1000_.jpg](saved_data\test1704661267.149294\happy\i00YzYyLTkxY2ItY2I3OWE3NDBmNDVmXkEyXkFqcGdeQXVyMjkwOTAyMDU._V1_FMjpg_UX1000_.jpg)                 | saved_data\test1704661267.149294\happy\i00YzYyLTkxY2ItY2I3OWE3NDBmNDVmXkEyXkFqcGdeQXVyMjkwOTAyMDU._V1_FMjpg_UX1000_.jpg | happy |
| ![if-you-recognize-these-signs-youre-a-naturally-happy-person.png](saved_data\test1704661267.149294\happy\if-you-recognize-these-signs-youre-a-naturally-happy-person.png)                                                   | saved_data\test1704661267.149294\happy\if-you-recognize-these-signs-youre-a-naturally-happy-person.png | happy |
| ![image.jpeg](saved_data\test1704661267.149294\happy\image.jpeg)                                                                                                                                                             | saved_data\test1704661267.149294\happy\image.jpeg | happy |
| ![image10.jpeg](saved_data\test1704661267.149294\happy\image10.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image10.jpeg | happy |
| ![image11.jpeg](saved_data\test1704661267.149294\happy\image11.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image11.jpeg | happy |
| ![image12.jpeg](saved_data\test1704661267.149294\happy\image12.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image12.jpeg | happy |
| ![image13.jpeg](saved_data\test1704661267.149294\happy\image13.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image13.jpeg | happy |
| ![image14.jpeg](saved_data\test1704661267.149294\happy\image14.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image14.jpeg | happy |
| ![image15.jpeg](saved_data\test1704661267.149294\happy\image15.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image15.jpeg | happy |
| ![image16.jpeg](saved_data\test1704661267.149294\happy\image16.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image16.jpeg | happy |
| ![image17.jpeg](saved_data\test1704661267.149294\happy\image17.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image17.jpeg | happy |
| ![image18.jpeg](saved_data\test1704661267.149294\happy\image18.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image18.jpeg | sad |
| ![image19.jpeg](saved_data\test1704661267.149294\happy\image19.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image19.jpeg | happy |
| ![image2.jpeg](saved_data\test1704661267.149294\happy\image2.jpeg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\image2.jpeg | sad |
| ![image20.jpeg](saved_data\test1704661267.149294\happy\image20.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image20.jpeg | happy |
| ![image21.jpeg](saved_data\test1704661267.149294\happy\image21.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image21.jpeg | happy |
| ![image22.jpeg](saved_data\test1704661267.149294\happy\image22.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image22.jpeg | happy |
| ![image23.jpeg](saved_data\test1704661267.149294\happy\image23.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image23.jpeg | happy |
| ![image24.jpeg](saved_data\test1704661267.149294\happy\image24.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image24.jpeg | happy |
| ![image25.jpeg](saved_data\test1704661267.149294\happy\image25.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image25.jpeg | happy |
| ![image26.jpeg](saved_data\test1704661267.149294\happy\image26.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image26.jpeg | happy |
| ![image27.jpeg](saved_data\test1704661267.149294\happy\image27.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image27.jpeg | happy |
| ![image28.jpeg](saved_data\test1704661267.149294\happy\image28.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image28.jpeg | happy |
| ![image29.jpeg](saved_data\test1704661267.149294\happy\image29.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image29.jpeg | happy |
| ![image3.jpeg](saved_data\test1704661267.149294\happy\image3.jpeg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\image3.jpeg | sad |
| ![image30.jpeg](saved_data\test1704661267.149294\happy\image30.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image30.jpeg | happy |
| ![image31.jpeg](saved_data\test1704661267.149294\happy\image31.jpeg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\image31.jpeg | happy |
| ![image32.png](saved_data\test1704661267.149294\happy\image32.png)                                                                                                                                                           | saved_data\test1704661267.149294\happy\image32.png | happy |
| ![image4.jpeg](saved_data\test1704661267.149294\happy\image4.jpeg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\image4.jpeg | happy |
| ![image5.jpeg](saved_data\test1704661267.149294\happy\image5.jpeg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\image5.jpeg | happy |
| ![image6.jpeg](saved_data\test1704661267.149294\happy\image6.jpeg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\image6.jpeg | happy |
| ![image7.jpeg](saved_data\test1704661267.149294\happy\image7.jpeg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\image7.jpeg | happy |
| ![image8.jpeg](saved_data\test1704661267.149294\happy\image8.jpeg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\image8.jpeg | happy |
| ![image9.jpeg](saved_data\test1704661267.149294\happy\image9.jpeg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\image9.jpeg | happy |
| ![images.jpg](saved_data\test1704661267.149294\happy\images.jpg)                                                                                                                                                             | saved_data\test1704661267.149294\happy\images.jpg | happy |
| ![images10.jpg](saved_data\test1704661267.149294\happy\images10.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images10.jpg | happy |
| ![images100.jpg](saved_data\test1704661267.149294\happy\images100.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images100.jpg | happy |
| ![images101.jpg](saved_data\test1704661267.149294\happy\images101.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images101.jpg | happy |
| ![images102.jpg](saved_data\test1704661267.149294\happy\images102.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images102.jpg | happy |
| ![images103.jpg](saved_data\test1704661267.149294\happy\images103.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images103.jpg | happy |
| ![images104.jpg](saved_data\test1704661267.149294\happy\images104.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images104.jpg | happy |
| ![images105.jpg](saved_data\test1704661267.149294\happy\images105.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images105.jpg | happy |
| ![images106.jpg](saved_data\test1704661267.149294\happy\images106.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images106.jpg | happy |
| ![images107.jpg](saved_data\test1704661267.149294\happy\images107.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images107.jpg | happy |
| ![images108.jpg](saved_data\test1704661267.149294\happy\images108.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images108.jpg | happy |
| ![images109.jpg](saved_data\test1704661267.149294\happy\images109.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images109.jpg | happy |
| ![images11.jpg](saved_data\test1704661267.149294\happy\images11.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images11.jpg | happy |
| ![images110.jpg](saved_data\test1704661267.149294\happy\images110.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images110.jpg | happy |
| ![images111.jpg](saved_data\test1704661267.149294\happy\images111.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images111.jpg | happy |
| ![images112.jpg](saved_data\test1704661267.149294\happy\images112.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images112.jpg | happy |
| ![images113.jpg](saved_data\test1704661267.149294\happy\images113.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images113.jpg | happy |
| ![images114.jpg](saved_data\test1704661267.149294\happy\images114.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images114.jpg | happy |
| ![images115.jpg](saved_data\test1704661267.149294\happy\images115.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images115.jpg | happy |
| ![images116.jpg](saved_data\test1704661267.149294\happy\images116.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images116.jpg | happy |
| ![images117.jpg](saved_data\test1704661267.149294\happy\images117.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images117.jpg | happy |
| ![images118.jpg](saved_data\test1704661267.149294\happy\images118.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images118.jpg | happy |
| ![images119.jpg](saved_data\test1704661267.149294\happy\images119.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images119.jpg | happy |
| ![images12.jpg](saved_data\test1704661267.149294\happy\images12.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images12.jpg | happy |
| ![images120.jpg](saved_data\test1704661267.149294\happy\images120.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images120.jpg | happy |
| ![images121.jpg](saved_data\test1704661267.149294\happy\images121.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images121.jpg | sad |
| ![images122.jpg](saved_data\test1704661267.149294\happy\images122.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images122.jpg | happy |
| ![images123.jpg](saved_data\test1704661267.149294\happy\images123.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images123.jpg | happy |
| ![images124.jpg](saved_data\test1704661267.149294\happy\images124.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images124.jpg | happy |
| ![images125.jpg](saved_data\test1704661267.149294\happy\images125.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images125.jpg | happy |
| ![images126.jpg](saved_data\test1704661267.149294\happy\images126.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images126.jpg | happy |
| ![images127.jpg](saved_data\test1704661267.149294\happy\images127.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images127.jpg | happy |
| ![images128.jpg](saved_data\test1704661267.149294\happy\images128.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images128.jpg | sad |
| ![images129.jpg](saved_data\test1704661267.149294\happy\images129.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images129.jpg | happy |
| ![images13.jpg](saved_data\test1704661267.149294\happy\images13.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images13.jpg | happy |
| ![images130.jpg](saved_data\test1704661267.149294\happy\images130.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images130.jpg | happy |
| ![images131.jpg](saved_data\test1704661267.149294\happy\images131.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images131.jpg | sad |
| ![images132.jpg](saved_data\test1704661267.149294\happy\images132.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images132.jpg | happy |
| ![images133.jpg](saved_data\test1704661267.149294\happy\images133.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images133.jpg | happy |
| ![images134.jpg](saved_data\test1704661267.149294\happy\images134.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images134.jpg | happy |
| ![images135.jpg](saved_data\test1704661267.149294\happy\images135.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images135.jpg | happy |
| ![images136.jpg](saved_data\test1704661267.149294\happy\images136.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images136.jpg | happy |
| ![images137.jpg](saved_data\test1704661267.149294\happy\images137.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images137.jpg | happy |
| ![images138.jpg](saved_data\test1704661267.149294\happy\images138.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images138.jpg | happy |
| ![images139.jpg](saved_data\test1704661267.149294\happy\images139.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images139.jpg | happy |
| ![images14.jpg](saved_data\test1704661267.149294\happy\images14.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images14.jpg | happy |
| ![images140.jpg](saved_data\test1704661267.149294\happy\images140.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images140.jpg | happy |
| ![images141.jpg](saved_data\test1704661267.149294\happy\images141.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images141.jpg | happy |
| ![images142.jpg](saved_data\test1704661267.149294\happy\images142.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images142.jpg | happy |
| ![images143.jpg](saved_data\test1704661267.149294\happy\images143.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images143.jpg | happy |
| ![images144.jpg](saved_data\test1704661267.149294\happy\images144.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images144.jpg | happy |
| ![images145.jpg](saved_data\test1704661267.149294\happy\images145.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images145.jpg | happy |
| ![images146.jpg](saved_data\test1704661267.149294\happy\images146.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images146.jpg | happy |
| ![images147.jpg](saved_data\test1704661267.149294\happy\images147.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images147.jpg | happy |
| ![images148.jpg](saved_data\test1704661267.149294\happy\images148.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images148.jpg | sad |
| ![images149.jpg](saved_data\test1704661267.149294\happy\images149.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images149.jpg | happy |
| ![images15.jpg](saved_data\test1704661267.149294\happy\images15.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images15.jpg | sad |
| ![images150.jpg](saved_data\test1704661267.149294\happy\images150.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images150.jpg | happy |
| ![images151.jpg](saved_data\test1704661267.149294\happy\images151.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images151.jpg | happy |
| ![images152.jpg](saved_data\test1704661267.149294\happy\images152.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images152.jpg | happy |
| ![images153.jpg](saved_data\test1704661267.149294\happy\images153.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images153.jpg | happy |
| ![images154.jpg](saved_data\test1704661267.149294\happy\images154.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images154.jpg | happy |
| ![images155.jpg](saved_data\test1704661267.149294\happy\images155.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images155.jpg | happy |
| ![images156.jpg](saved_data\test1704661267.149294\happy\images156.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images156.jpg | sad |
| ![images157.jpg](saved_data\test1704661267.149294\happy\images157.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images157.jpg | happy |
| ![images158.jpg](saved_data\test1704661267.149294\happy\images158.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images158.jpg | happy |
| ![images159.jpg](saved_data\test1704661267.149294\happy\images159.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images159.jpg | happy |
| ![images16.jpg](saved_data\test1704661267.149294\happy\images16.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images16.jpg | happy |
| ![images160.jpg](saved_data\test1704661267.149294\happy\images160.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images160.jpg | happy |
| ![images161.jpg](saved_data\test1704661267.149294\happy\images161.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images161.jpg | happy |
| ![images162.jpg](saved_data\test1704661267.149294\happy\images162.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images162.jpg | happy |
| ![images163.jpg](saved_data\test1704661267.149294\happy\images163.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images163.jpg | happy |
| ![images164.jpg](saved_data\test1704661267.149294\happy\images164.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images164.jpg | happy |
| ![images165.jpg](saved_data\test1704661267.149294\happy\images165.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images165.jpg | happy |
| ![images166.jpg](saved_data\test1704661267.149294\happy\images166.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images166.jpg | happy |
| ![images167.jpg](saved_data\test1704661267.149294\happy\images167.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images167.jpg | happy |
| ![images168.jpg](saved_data\test1704661267.149294\happy\images168.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images168.jpg | happy |
| ![images169.jpg](saved_data\test1704661267.149294\happy\images169.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images169.jpg | happy |
| ![images17.jpg](saved_data\test1704661267.149294\happy\images17.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images17.jpg | happy |
| ![images170.jpg](saved_data\test1704661267.149294\happy\images170.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images170.jpg | happy |
| ![images171.jpg](saved_data\test1704661267.149294\happy\images171.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images171.jpg | happy |
| ![images172.jpg](saved_data\test1704661267.149294\happy\images172.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images172.jpg | happy |
| ![images173.jpg](saved_data\test1704661267.149294\happy\images173.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images173.jpg | happy |
| ![images174.jpg](saved_data\test1704661267.149294\happy\images174.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images174.jpg | happy |
| ![images175.jpg](saved_data\test1704661267.149294\happy\images175.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images175.jpg | happy |
| ![images176.jpg](saved_data\test1704661267.149294\happy\images176.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images176.jpg | happy |
| ![images177.jpg](saved_data\test1704661267.149294\happy\images177.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images177.jpg | happy |
| ![images178.jpg](saved_data\test1704661267.149294\happy\images178.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images178.jpg | happy |
| ![images179.jpg](saved_data\test1704661267.149294\happy\images179.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images179.jpg | happy |
| ![images18.jpg](saved_data\test1704661267.149294\happy\images18.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images18.jpg | happy |
| ![images180.jpg](saved_data\test1704661267.149294\happy\images180.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images180.jpg | happy |
| ![images181.jpg](saved_data\test1704661267.149294\happy\images181.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images181.jpg | happy |
| ![images182.jpg](saved_data\test1704661267.149294\happy\images182.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images182.jpg | happy |
| ![images183.jpg](saved_data\test1704661267.149294\happy\images183.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images183.jpg | happy |
| ![images184.jpg](saved_data\test1704661267.149294\happy\images184.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images184.jpg | happy |
| ![images185.jpg](saved_data\test1704661267.149294\happy\images185.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images185.jpg | sad |
| ![images186.jpg](saved_data\test1704661267.149294\happy\images186.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images186.jpg | sad |
| ![images187.jpg](saved_data\test1704661267.149294\happy\images187.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images187.jpg | happy |
| ![images188.jpg](saved_data\test1704661267.149294\happy\images188.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images188.jpg | happy |
| ![images189.jpg](saved_data\test1704661267.149294\happy\images189.jpg)                                                                                                                                                       | saved_data\test1704661267.149294\happy\images189.jpg | happy |
| ![images19.jpg](saved_data\test1704661267.149294\happy\images19.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images19.jpg | happy |
| ![images2.jpg](saved_data\test1704661267.149294\happy\images2.jpg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\images2.jpg | happy |
| ![images20.jpg](saved_data\test1704661267.149294\happy\images20.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images20.jpg | happy |
| ![images21.jpg](saved_data\test1704661267.149294\happy\images21.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images21.jpg | happy |
| ![images22.jpg](saved_data\test1704661267.149294\happy\images22.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images22.jpg | happy |
| ![images23.jpg](saved_data\test1704661267.149294\happy\images23.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images23.jpg | happy |
| ![images24.jpg](saved_data\test1704661267.149294\happy\images24.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images24.jpg | sad |
| ![images25.jpg](saved_data\test1704661267.149294\happy\images25.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images25.jpg | happy |
| ![images26.jpg](saved_data\test1704661267.149294\happy\images26.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images26.jpg | happy |
| ![images27.jpg](saved_data\test1704661267.149294\happy\images27.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images27.jpg | happy |
| ![images28.jpg](saved_data\test1704661267.149294\happy\images28.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images28.jpg | happy |
| ![images29.jpg](saved_data\test1704661267.149294\happy\images29.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images29.jpg | happy |
| ![images3.jpg](saved_data\test1704661267.149294\happy\images3.jpg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\images3.jpg | happy |
| ![images30.jpg](saved_data\test1704661267.149294\happy\images30.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images30.jpg | happy |
| ![images31.jpg](saved_data\test1704661267.149294\happy\images31.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images31.jpg | happy |
| ![images32.jpg](saved_data\test1704661267.149294\happy\images32.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images32.jpg | sad |
| ![images33.jpg](saved_data\test1704661267.149294\happy\images33.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images33.jpg | sad |
| ![images34.jpg](saved_data\test1704661267.149294\happy\images34.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images34.jpg | happy |
| ![images35.jpg](saved_data\test1704661267.149294\happy\images35.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images35.jpg | happy |
| ![images36.jpg](saved_data\test1704661267.149294\happy\images36.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images36.jpg | happy |
| ![images37.jpg](saved_data\test1704661267.149294\happy\images37.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images37.jpg | happy |
| ![images38.jpg](saved_data\test1704661267.149294\happy\images38.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images38.jpg | happy |
| ![images39.jpg](saved_data\test1704661267.149294\happy\images39.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images39.jpg | happy |
| ![images4.jpg](saved_data\test1704661267.149294\happy\images4.jpg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\images4.jpg | happy |
| ![images40.jpg](saved_data\test1704661267.149294\happy\images40.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images40.jpg | happy |
| ![images41.jpg](saved_data\test1704661267.149294\happy\images41.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images41.jpg | happy |
| ![images42.jpg](saved_data\test1704661267.149294\happy\images42.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images42.jpg | happy |
| ![images43.jpg](saved_data\test1704661267.149294\happy\images43.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images43.jpg | happy |
| ![images44.jpg](saved_data\test1704661267.149294\happy\images44.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images44.jpg | happy |
| ![images45.jpg](saved_data\test1704661267.149294\happy\images45.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images45.jpg | happy |
| ![images46.jpg](saved_data\test1704661267.149294\happy\images46.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images46.jpg | happy |
| ![images47.jpg](saved_data\test1704661267.149294\happy\images47.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images47.jpg | happy |
| ![images48.jpg](saved_data\test1704661267.149294\happy\images48.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images48.jpg | happy |
| ![images49.jpg](saved_data\test1704661267.149294\happy\images49.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images49.jpg | happy |
| ![images5.jpg](saved_data\test1704661267.149294\happy\images5.jpg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\images5.jpg | happy |
| ![images50.jpg](saved_data\test1704661267.149294\happy\images50.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images50.jpg | happy |
| ![images51.jpg](saved_data\test1704661267.149294\happy\images51.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images51.jpg | happy |
| ![images52.jpg](saved_data\test1704661267.149294\happy\images52.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images52.jpg | happy |
| ![images53.jpg](saved_data\test1704661267.149294\happy\images53.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images53.jpg | happy |
| ![images54.jpg](saved_data\test1704661267.149294\happy\images54.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images54.jpg | happy |
| ![images55.jpg](saved_data\test1704661267.149294\happy\images55.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images55.jpg | happy |
| ![images56.jpg](saved_data\test1704661267.149294\happy\images56.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images56.jpg | happy |
| ![images57.jpg](saved_data\test1704661267.149294\happy\images57.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images57.jpg | happy |
| ![images58.jpg](saved_data\test1704661267.149294\happy\images58.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images58.jpg | happy |
| ![images59.jpg](saved_data\test1704661267.149294\happy\images59.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images59.jpg | happy |
| ![images6.jpg](saved_data\test1704661267.149294\happy\images6.jpg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\images6.jpg | happy |
| ![images60.jpg](saved_data\test1704661267.149294\happy\images60.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images60.jpg | happy |
| ![images61.jpg](saved_data\test1704661267.149294\happy\images61.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images61.jpg | happy |
| ![images62.jpg](saved_data\test1704661267.149294\happy\images62.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images62.jpg | happy |
| ![images63.jpg](saved_data\test1704661267.149294\happy\images63.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images63.jpg | happy |
| ![images64.jpg](saved_data\test1704661267.149294\happy\images64.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images64.jpg | happy |
| ![images65.jpg](saved_data\test1704661267.149294\happy\images65.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images65.jpg | happy |
| ![images66.jpg](saved_data\test1704661267.149294\happy\images66.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images66.jpg | happy |
| ![images67.jpg](saved_data\test1704661267.149294\happy\images67.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images67.jpg | happy |
| ![images68.jpg](saved_data\test1704661267.149294\happy\images68.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images68.jpg | happy |
| ![images69.jpg](saved_data\test1704661267.149294\happy\images69.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images69.jpg | happy |
| ![images7.jpg](saved_data\test1704661267.149294\happy\images7.jpg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\images7.jpg | happy |
| ![images70.jpg](saved_data\test1704661267.149294\happy\images70.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images70.jpg | happy |
| ![images71.jpg](saved_data\test1704661267.149294\happy\images71.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images71.jpg | happy |
| ![images72.jpg](saved_data\test1704661267.149294\happy\images72.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images72.jpg | happy |
| ![images73.jpg](saved_data\test1704661267.149294\happy\images73.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images73.jpg | happy |
| ![images74.jpg](saved_data\test1704661267.149294\happy\images74.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images74.jpg | happy |
| ![images75.jpg](saved_data\test1704661267.149294\happy\images75.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images75.jpg | sad |
| ![images76.jpg](saved_data\test1704661267.149294\happy\images76.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images76.jpg | happy |
| ![images77.jpg](saved_data\test1704661267.149294\happy\images77.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images77.jpg | happy |
| ![images78.jpg](saved_data\test1704661267.149294\happy\images78.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images78.jpg | happy |
| ![images79.jpg](saved_data\test1704661267.149294\happy\images79.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images79.jpg | happy |
| ![images8.jpg](saved_data\test1704661267.149294\happy\images8.jpg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\images8.jpg | sad |
| ![images80.jpg](saved_data\test1704661267.149294\happy\images80.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images80.jpg | sad |
| ![images81.jpg](saved_data\test1704661267.149294\happy\images81.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images81.jpg | happy |
| ![images82.jpg](saved_data\test1704661267.149294\happy\images82.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images82.jpg | happy |
| ![images83.jpg](saved_data\test1704661267.149294\happy\images83.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images83.jpg | happy |
| ![images84.jpg](saved_data\test1704661267.149294\happy\images84.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images84.jpg | happy |
| ![images85.jpg](saved_data\test1704661267.149294\happy\images85.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images85.jpg | happy |
| ![images86.jpg](saved_data\test1704661267.149294\happy\images86.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images86.jpg | sad |
| ![images87.jpg](saved_data\test1704661267.149294\happy\images87.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images87.jpg | happy |
| ![images88.jpg](saved_data\test1704661267.149294\happy\images88.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images88.jpg | happy |
| ![images89.jpg](saved_data\test1704661267.149294\happy\images89.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images89.jpg | happy |
| ![images9.jpg](saved_data\test1704661267.149294\happy\images9.jpg)                                                                                                                                                           | saved_data\test1704661267.149294\happy\images9.jpg | happy |
| ![images90.jpg](saved_data\test1704661267.149294\happy\images90.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images90.jpg | happy |
| ![images91.jpg](saved_data\test1704661267.149294\happy\images91.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images91.jpg | happy |
| ![images92.jpg](saved_data\test1704661267.149294\happy\images92.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images92.jpg | happy |
| ![images93.jpg](saved_data\test1704661267.149294\happy\images93.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images93.jpg | happy |
| ![images94.jpg](saved_data\test1704661267.149294\happy\images94.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images94.jpg | happy |
| ![images95.jpg](saved_data\test1704661267.149294\happy\images95.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images95.jpg | sad |
| ![images96.jpg](saved_data\test1704661267.149294\happy\images96.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images96.jpg | happy |
| ![images97.jpg](saved_data\test1704661267.149294\happy\images97.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images97.jpg | happy |
| ![images98.jpg](saved_data\test1704661267.149294\happy\images98.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images98.jpg | sad |
| ![images99.jpg](saved_data\test1704661267.149294\happy\images99.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\images99.jpg | happy |
| ![image_jumpstory-download20230421-155938_7a7b427.jpg](saved_data\test1704661267.149294\happy\image_jumpstory-download20230421-155938_7a7b427.jpg)                                                                           | saved_data\test1704661267.149294\happy\image_jumpstory-download20230421-155938_7a7b427.jpg | happy |
| ![ipsos-global-advisor-happiness-2022-opti.jpg](saved_data\test1704661267.149294\happy\ipsos-global-advisor-happiness-2022-opti.jpg)                                                                                         | saved_data\test1704661267.149294\happy\ipsos-global-advisor-happiness-2022-opti.jpg | sad |
| ![jumping-and-dancing-happy-people-positive-emotions-set-illustration-free-vector.jpg](saved_data\test1704661267.149294\happy\jumping-and-dancing-happy-people-positive-emotions-set-illustration-free-vector.jpg)           | saved_data\test1704661267.149294\happy\jumping-and-dancing-happy-people-positive-emotions-set-illustration-free-vector.jpg | happy |
| ![maxresdefault.jpg](saved_data\test1704661267.149294\happy\maxresdefault.jpg)                                                                                                                                               | saved_data\test1704661267.149294\happy\maxresdefault.jpg | happy |
| ![maxresdefault2.jpg](saved_data\test1704661267.149294\happy\maxresdefault2.jpg)                                                                                                                                             | saved_data\test1704661267.149294\happy\maxresdefault2.jpg | happy |
| ![MV5BMGI1NzJiMWUtMmYwMS00MDllLWJkYWYtNDVhOTY0M2U5ODdmXkEyXkFqcGdeQXVyMjA0MzYwMDY._V1_.jpg](saved_data\test1704661267.149294\happy\MV5BMGI1NzJiMWUtMmYwMS00MDllLWJkYWYtNDVhOTY0M2U5ODdmXkEyXkFqcGdeQXVyMjA0MzYwMDY._V1_.jpg) | saved_data\test1704661267.149294\happy\MV5BMGI1NzJiMWUtMmYwMS00MDllLWJkYWYtNDVhOTY0M2U5ODdmXkEyXkFqcGdeQXVyMjA0MzYwMDY._V1_.jpg | happy |
| ![o-HAPPY-facebook.jpg](saved_data\test1704661267.149294\happy\o-HAPPY-facebook.jpg)                                                                                                                                         | saved_data\test1704661267.149294\happy\o-HAPPY-facebook.jpg | sad |
| ![o-happy-old-people-facebook.jpg](saved_data\test1704661267.149294\happy\o-happy-old-people-facebook.jpg)                                                                                                                   | saved_data\test1704661267.149294\happy\o-happy-old-people-facebook.jpg | happy |
| ![p074953m.jpg](saved_data\test1704661267.149294\happy\p074953m.jpg)                                                                                                                                                         | saved_data\test1704661267.149294\happy\p074953m.jpg | happy |
| ![png-clipart-happiness-graphy-smile-happy-people-love-photography.png](saved_data\test1704661267.149294\happy\png-clipart-happiness-graphy-smile-happy-people-love-photography.png)                                         | saved_data\test1704661267.149294\happy\png-clipart-happiness-graphy-smile-happy-people-love-photography.png | happy |
| ![Screaming-Happy-Woman-THe-Trent.jpg](saved_data\test1704661267.149294\happy\Screaming-Happy-Woman-THe-Trent.jpg)                                                                                                           | saved_data\test1704661267.149294\happy\Screaming-Happy-Woman-THe-Trent.jpg | happy |
| ![Screen-Shot-2012-10-23-at-12.57.22-PM.png](saved_data\test1704661267.149294\happy\Screen-Shot-2012-10-23-at-12.57.22-PM.png)                                                                                               | saved_data\test1704661267.149294\happy\Screen-Shot-2012-10-23-at-12.57.22-PM.png | happy |
| ![smile.woman_.jpg](saved_data\test1704661267.149294\happy\smile.woman_.jpg)                                                                                                                                                 | saved_data\test1704661267.149294\happy\smile.woman_.jpg | happy |
| ![Successful-year.jpg](saved_data\test1704661267.149294\happy\Successful-year.jpg)                                                                                                                                           | saved_data\test1704661267.149294\happy\Successful-year.jpg | happy |
| ![Super-Happy-People-yay.jpg](saved_data\test1704661267.149294\happy\Super-Happy-People-yay.jpg)                                                                                                                             | saved_data\test1704661267.149294\happy\Super-Happy-People-yay.jpg | sad |
| ![traitshappypeople.jpg](saved_data\test1704661267.149294\happy\traitshappypeople.jpg)                                                                                                                                       | saved_data\test1704661267.149294\happy\traitshappypeople.jpg | sad |
| ![Travis-Bradberry-Happy.jpg](saved_data\test1704661267.149294\happy\Travis-Bradberry-Happy.jpg)                                                                                                                             | saved_data\test1704661267.149294\happy\Travis-Bradberry-Happy.jpg | happy |
| ![very-happy-people.jpg](saved_data\test1704661267.149294\happy\very-happy-people.jpg)                                                                                                                                       | saved_data\test1704661267.149294\happy\very-happy-people.jpg | happy |
| ![web3-happy-people-outside-smile-sun-nature-eduardo-dutra-620857-unsplash.jpg](saved_data\test1704661267.149294\happy\web3-happy-people-outside-smile-sun-nature-eduardo-dutra-620857-unsplash.jpg)                         | saved_data\test1704661267.149294\happy\web3-happy-people-outside-smile-sun-nature-eduardo-dutra-620857-unsplash.jpg | happy |
| ![young-and-happy-people-vector-15114154.jpg](saved_data\test1704661267.149294\happy\young-and-happy-people-vector-15114154.jpg)                                                                                             | saved_data\test1704661267.149294\happy\young-and-happy-people-vector-15114154.jpg | happy |
| ![_happy_jumping_on_beach-40815.jpg](saved_data\test1704661267.149294\happy\_happy_jumping_on_beach-40815.jpg)                                                                                                               | saved_data\test1704661267.149294\happy\_happy_jumping_on_beach-40815.jpg | happy |

Number of correct predicted images 280 from a total number of 307 images.



Predictions for sad test dataset:

| Image                                                                                                                                                  | Image path                                                                                   | Predicted class |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|-----------------|
| ![-unhappy-miss-good-chance-dressed-casually-isolated-yellow-wall_273609-37534.jpg](saved_data\test1704661267.149294\sad\-unhappy-miss-good-chance-dressed-casually-isolated-yellow-wall_273609-37534.jpg) | saved_data\test1704661267.149294\sad\-unhappy-miss-good-chance-dressed-casually-isolated-yellow-wall_273609-37534.jpg | sad |
| ![1000_F_127508805_RlWmvDSE5FE4TPc53k4ld0NryWxeLsad.jpg](saved_data\test1704661267.149294\sad\1000_F_127508805_RlWmvDSE5FE4TPc53k4ld0NryWxeLsad.jpg) | saved_data\test1704661267.149294\sad\1000_F_127508805_RlWmvDSE5FE4TPc53k4ld0NryWxeLsad.jpg | sad |
| ![1000_F_233515059_A3JEmjWEgLcwOAc7aNxa6k3SDXjvvBgv.jpg](saved_data\test1704661267.149294\sad\1000_F_233515059_A3JEmjWEgLcwOAc7aNxa6k3SDXjvvBgv.jpg) | saved_data\test1704661267.149294\sad\1000_F_233515059_A3JEmjWEgLcwOAc7aNxa6k3SDXjvvBgv.jpg | sad |
| ![1000_F_452957465_ZDlbGO5mwQ4LisGrusfhuFwYDG2by4lc.jpg](saved_data\test1704661267.149294\sad\1000_F_452957465_ZDlbGO5mwQ4LisGrusfhuFwYDG2by4lc.jpg) | saved_data\test1704661267.149294\sad\1000_F_452957465_ZDlbGO5mwQ4LisGrusfhuFwYDG2by4lc.jpg | sad |
| ![1000_F_584080922_GDrrJBOpwC2AOvbDIdPfPcxEF0RrTRgk.jpg](saved_data\test1704661267.149294\sad\1000_F_584080922_GDrrJBOpwC2AOvbDIdPfPcxEF0RrTRgk.jpg) | saved_data\test1704661267.149294\sad\1000_F_584080922_GDrrJBOpwC2AOvbDIdPfPcxEF0RrTRgk.jpg | sad |
| ![12165734.jpg](saved_data\test1704661267.149294\sad\12165734.jpg) | saved_data\test1704661267.149294\sad\12165734.jpg | sad |
| ![141203-depression-stock.jpg](saved_data\test1704661267.149294\sad\141203-depression-stock.jpg) | saved_data\test1704661267.149294\sad\141203-depression-stock.jpg | sad |
| ![285-2854909_people-boy-k-sad-person-cartoon-transparent.png](saved_data\test1704661267.149294\sad\285-2854909_people-boy-k-sad-person-cartoon-transparent.png) | saved_data\test1704661267.149294\sad\285-2854909_people-boy-k-sad-person-cartoon-transparent.png | sad |
| ![353397713.jpg](saved_data\test1704661267.149294\sad\353397713.jpg) | saved_data\test1704661267.149294\sad\353397713.jpg | sad |
| ![360_F_548848756_IlV9Y9HV8chb6mSuc3PBamYRT9gIn8Vo.jpg](saved_data\test1704661267.149294\sad\360_F_548848756_IlV9Y9HV8chb6mSuc3PBamYRT9gIn8Vo.jpg) | saved_data\test1704661267.149294\sad\360_F_548848756_IlV9Y9HV8chb6mSuc3PBamYRT9gIn8Vo.jpg | sad |
| ![360_F_573380015_l5YdjSZuJqET7UWOHBVMhzT7J63O8hPq.jpg](saved_data\test1704661267.149294\sad\360_F_573380015_l5YdjSZuJqET7UWOHBVMhzT7J63O8hPq.jpg) | saved_data\test1704661267.149294\sad\360_F_573380015_l5YdjSZuJqET7UWOHBVMhzT7J63O8hPq.jpg | sad |
| ![360_F_601507482_RbV0Vk2KSg72LkjkSZBJNpmxu6Y4Hdzw.jpg](saved_data\test1704661267.149294\sad\360_F_601507482_RbV0Vk2KSg72LkjkSZBJNpmxu6Y4Hdzw.jpg) | saved_data\test1704661267.149294\sad\360_F_601507482_RbV0Vk2KSg72LkjkSZBJNpmxu6Y4Hdzw.jpg | sad |
| ![39843138-sad-man.jpg](saved_data\test1704661267.149294\sad\39843138-sad-man.jpg) | saved_data\test1704661267.149294\sad\39843138-sad-man.jpg | sad |
| ![5360f7e3f9a01bb1aa10654514442436.500x500x1.jpg](saved_data\test1704661267.149294\sad\5360f7e3f9a01bb1aa10654514442436.500x500x1.jpg) | saved_data\test1704661267.149294\sad\5360f7e3f9a01bb1aa10654514442436.500x500x1.jpg | sad |
| ![5acf9ed1146e711e008b46d7.jpg](saved_data\test1704661267.149294\sad\5acf9ed1146e711e008b46d7.jpg) | saved_data\test1704661267.149294\sad\5acf9ed1146e711e008b46d7.jpg | sad |
| ![644019f021b5ff36764928ed_people-icon-sad.png](saved_data\test1704661267.149294\sad\644019f021b5ff36764928ed_people-icon-sad.png) | saved_data\test1704661267.149294\sad\644019f021b5ff36764928ed_people-icon-sad.png | sad |
| ![73705bd7debb66c2afc780a22c223804.jpg](saved_data\test1704661267.149294\sad\73705bd7debb66c2afc780a22c223804.jpg) | saved_data\test1704661267.149294\sad\73705bd7debb66c2afc780a22c223804.jpg | sad |
| ![7RNXwSxCAKL8vGtXG2ZkyD-1200-80.jpg](saved_data\test1704661267.149294\sad\7RNXwSxCAKL8vGtXG2ZkyD-1200-80.jpg) | saved_data\test1704661267.149294\sad\7RNXwSxCAKL8vGtXG2ZkyD-1200-80.jpg | sad |
| ![8iAb9k4aT.jpg](saved_data\test1704661267.149294\sad\8iAb9k4aT.jpg) | saved_data\test1704661267.149294\sad\8iAb9k4aT.jpg | sad |
| ![960x0.jpg](saved_data\test1704661267.149294\sad\960x0.jpg) | saved_data\test1704661267.149294\sad\960x0.jpg | sad |
| ![a-lonely-and-sad-person-sitting-on-a-bench-created-with-generative-ai-technology-photo.jpg](saved_data\test1704661267.149294\sad\a-lonely-and-sad-person-sitting-on-a-bench-created-with-generative-ai-technology-photo.jpg) | saved_data\test1704661267.149294\sad\a-lonely-and-sad-person-sitting-on-a-bench-created-with-generative-ai-technology-photo.jpg | sad |
| ![ANLem4bnFq0ysIBZ60imcOQ2_mzu7Z802BGAclqe63nws64-c-mo.jpg](saved_data\test1704661267.149294\sad\ANLem4bnFq0ysIBZ60imcOQ2_mzu7Z802BGAclqe63nws64-c-mo.jpg) | saved_data\test1704661267.149294\sad\ANLem4bnFq0ysIBZ60imcOQ2_mzu7Z802BGAclqe63nws64-c-mo.jpg | happy |
| ![anxious-man-indoors-front-view_23-2149729600.jpg](saved_data\test1704661267.149294\sad\anxious-man-indoors-front-view_23-2149729600.jpg) | saved_data\test1704661267.149294\sad\anxious-man-indoors-front-view_23-2149729600.jpg | sad |
| ![b2ap3_large_happy-sad-unsplash-850x575.jpg](saved_data\test1704661267.149294\sad\b2ap3_large_happy-sad-unsplash-850x575.jpg) | saved_data\test1704661267.149294\sad\b2ap3_large_happy-sad-unsplash-850x575.jpg | sad |
| ![CC_HE_1221887081_SituationalDepression.jpg](saved_data\test1704661267.149294\sad\CC_HE_1221887081_SituationalDepression.jpg) | saved_data\test1704661267.149294\sad\CC_HE_1221887081_SituationalDepression.jpg | sad |
| ![crying-at-work.jpg](saved_data\test1704661267.149294\sad\crying-at-work.jpg) | saved_data\test1704661267.149294\sad\crying-at-work.jpg | sad |
| ![dark-depression-mood-people-wallpaper-preview.jpg](saved_data\test1704661267.149294\sad\dark-depression-mood-people-wallpaper-preview.jpg) | saved_data\test1704661267.149294\sad\dark-depression-mood-people-wallpaper-preview.jpg | sad |
| ![de2a0f4d137d0aeb839d43a2ea9b6c72.jpg](saved_data\test1704661267.149294\sad\de2a0f4d137d0aeb839d43a2ea9b6c72.jpg) | saved_data\test1704661267.149294\sad\de2a0f4d137d0aeb839d43a2ea9b6c72.jpg | sad |
| ![DealingwithDepressionwithoutMedication-1.jpg](saved_data\test1704661267.149294\sad\DealingwithDepressionwithoutMedication-1.jpg) | saved_data\test1704661267.149294\sad\DealingwithDepressionwithoutMedication-1.jpg | sad |
| ![depressed-person-standing-alone-bench_23-2150761438.jpg](saved_data\test1704661267.149294\sad\depressed-person-standing-alone-bench_23-2150761438.jpg) | saved_data\test1704661267.149294\sad\depressed-person-standing-alone-bench_23-2150761438.jpg | sad |
| ![depression-1020x680.jpg](saved_data\test1704661267.149294\sad\depression-1020x680.jpg) | saved_data\test1704661267.149294\sad\depression-1020x680.jpg | sad |
| ![depression-sad-mood-sorrow-dark-people-wallpaper-7.jpg](saved_data\test1704661267.149294\sad\depression-sad-mood-sorrow-dark-people-wallpaper-7.jpg) | saved_data\test1704661267.149294\sad\depression-sad-mood-sorrow-dark-people-wallpaper-7.jpg | sad |
| ![Depression-Vs-Sadness-Are-You-Just-Sad-Or-Depressed-2020-960x640.jpg](saved_data\test1704661267.149294\sad\Depression-Vs-Sadness-Are-You-Just-Sad-Or-Depressed-2020-960x640.jpg) | saved_data\test1704661267.149294\sad\Depression-Vs-Sadness-Are-You-Just-Sad-Or-Depressed-2020-960x640.jpg | sad |
| ![dreamstime_s_101440985.jpg](saved_data\test1704661267.149294\sad\dreamstime_s_101440985.jpg) | saved_data\test1704661267.149294\sad\dreamstime_s_101440985.jpg | sad |
| ![front-view-sad-arfo-american-man_23-2148398423.jpg](saved_data\test1704661267.149294\sad\front-view-sad-arfo-american-man_23-2148398423.jpg) | saved_data\test1704661267.149294\sad\front-view-sad-arfo-american-man_23-2148398423.jpg | sad |
| ![girl-love-boyfriend-divorce-background_1150-1511.jpg](saved_data\test1704661267.149294\sad\girl-love-boyfriend-divorce-background_1150-1511.jpg) | saved_data\test1704661267.149294\sad\girl-love-boyfriend-divorce-background_1150-1511.jpg | sad |
| ![hqdefault.jpg](saved_data\test1704661267.149294\sad\hqdefault.jpg) | saved_data\test1704661267.149294\sad\hqdefault.jpg | sad |
| ![image-20160914-4963-19knfh1.jpg](saved_data\test1704661267.149294\sad\image-20160914-4963-19knfh1.jpg) | saved_data\test1704661267.149294\sad\image-20160914-4963-19knfh1.jpg | sad |
| ![image-asset.jpeg](saved_data\test1704661267.149294\sad\image-asset.jpeg) | saved_data\test1704661267.149294\sad\image-asset.jpeg | sad |
| ![image.jpeg](saved_data\test1704661267.149294\sad\image.jpeg) | saved_data\test1704661267.149294\sad\image.jpeg | sad |
| ![image10.jpeg](saved_data\test1704661267.149294\sad\image10.jpeg) | saved_data\test1704661267.149294\sad\image10.jpeg | sad |
| ![image11.jpeg](saved_data\test1704661267.149294\sad\image11.jpeg) | saved_data\test1704661267.149294\sad\image11.jpeg | sad |
| ![image12.jpeg](saved_data\test1704661267.149294\sad\image12.jpeg) | saved_data\test1704661267.149294\sad\image12.jpeg | sad |
| ![image13.jpeg](saved_data\test1704661267.149294\sad\image13.jpeg) | saved_data\test1704661267.149294\sad\image13.jpeg | sad |
| ![image14.jpeg](saved_data\test1704661267.149294\sad\image14.jpeg) | saved_data\test1704661267.149294\sad\image14.jpeg | sad |
| ![image15.jpeg](saved_data\test1704661267.149294\sad\image15.jpeg) | saved_data\test1704661267.149294\sad\image15.jpeg | sad |
| ![image16.jpeg](saved_data\test1704661267.149294\sad\image16.jpeg) | saved_data\test1704661267.149294\sad\image16.jpeg | sad |
| ![image17.jpeg](saved_data\test1704661267.149294\sad\image17.jpeg) | saved_data\test1704661267.149294\sad\image17.jpeg | sad |
| ![image18.jpeg](saved_data\test1704661267.149294\sad\image18.jpeg) | saved_data\test1704661267.149294\sad\image18.jpeg | sad |
| ![image19.jpeg](saved_data\test1704661267.149294\sad\image19.jpeg) | saved_data\test1704661267.149294\sad\image19.jpeg | sad |
| ![image2.jpeg](saved_data\test1704661267.149294\sad\image2.jpeg) | saved_data\test1704661267.149294\sad\image2.jpeg | sad |
| ![image20.jpeg](saved_data\test1704661267.149294\sad\image20.jpeg) | saved_data\test1704661267.149294\sad\image20.jpeg | sad |
| ![image21.jpeg](saved_data\test1704661267.149294\sad\image21.jpeg) | saved_data\test1704661267.149294\sad\image21.jpeg | sad |
| ![image22.jpeg](saved_data\test1704661267.149294\sad\image22.jpeg) | saved_data\test1704661267.149294\sad\image22.jpeg | sad |
| ![image23.jpeg](saved_data\test1704661267.149294\sad\image23.jpeg) | saved_data\test1704661267.149294\sad\image23.jpeg | sad |
| ![image24.jpeg](saved_data\test1704661267.149294\sad\image24.jpeg) | saved_data\test1704661267.149294\sad\image24.jpeg | sad |
| ![image25.jpeg](saved_data\test1704661267.149294\sad\image25.jpeg) | saved_data\test1704661267.149294\sad\image25.jpeg | sad |
| ![image26.jpeg](saved_data\test1704661267.149294\sad\image26.jpeg) | saved_data\test1704661267.149294\sad\image26.jpeg | sad |
| ![image27.jpeg](saved_data\test1704661267.149294\sad\image27.jpeg) | saved_data\test1704661267.149294\sad\image27.jpeg | sad |
| ![image28.jpeg](saved_data\test1704661267.149294\sad\image28.jpeg) | saved_data\test1704661267.149294\sad\image28.jpeg | sad |
| ![image29.jpeg](saved_data\test1704661267.149294\sad\image29.jpeg) | saved_data\test1704661267.149294\sad\image29.jpeg | sad |
| ![image3.jpeg](saved_data\test1704661267.149294\sad\image3.jpeg) | saved_data\test1704661267.149294\sad\image3.jpeg | sad |
| ![image30.jpeg](saved_data\test1704661267.149294\sad\image30.jpeg) | saved_data\test1704661267.149294\sad\image30.jpeg | sad |
| ![image31.jpeg](saved_data\test1704661267.149294\sad\image31.jpeg) | saved_data\test1704661267.149294\sad\image31.jpeg | sad |
| ![image32.png](saved_data\test1704661267.149294\sad\image32.png) | saved_data\test1704661267.149294\sad\image32.png | sad |
| ![image4.jpeg](saved_data\test1704661267.149294\sad\image4.jpeg) | saved_data\test1704661267.149294\sad\image4.jpeg | sad |
| ![image5.jpeg](saved_data\test1704661267.149294\sad\image5.jpeg) | saved_data\test1704661267.149294\sad\image5.jpeg | happy |
| ![image6.jpeg](saved_data\test1704661267.149294\sad\image6.jpeg) | saved_data\test1704661267.149294\sad\image6.jpeg | sad |
| ![image7.jpeg](saved_data\test1704661267.149294\sad\image7.jpeg) | saved_data\test1704661267.149294\sad\image7.jpeg | sad |
| ![image8.jpeg](saved_data\test1704661267.149294\sad\image8.jpeg) | saved_data\test1704661267.149294\sad\image8.jpeg | sad |
| ![image9.jpeg](saved_data\test1704661267.149294\sad\image9.jpeg) | saved_data\test1704661267.149294\sad\image9.jpeg | sad |
| ![images.jpg](saved_data\test1704661267.149294\sad\images.jpg) | saved_data\test1704661267.149294\sad\images.jpg | sad |
| ![images10.jpg](saved_data\test1704661267.149294\sad\images10.jpg) | saved_data\test1704661267.149294\sad\images10.jpg | sad |
| ![images100.jpg](saved_data\test1704661267.149294\sad\images100.jpg) | saved_data\test1704661267.149294\sad\images100.jpg | sad |
| ![images101.jpg](saved_data\test1704661267.149294\sad\images101.jpg) | saved_data\test1704661267.149294\sad\images101.jpg | sad |
| ![images102.jpg](saved_data\test1704661267.149294\sad\images102.jpg) | saved_data\test1704661267.149294\sad\images102.jpg | sad |
| ![images103.jpg](saved_data\test1704661267.149294\sad\images103.jpg) | saved_data\test1704661267.149294\sad\images103.jpg | sad |
| ![images104.jpg](saved_data\test1704661267.149294\sad\images104.jpg) | saved_data\test1704661267.149294\sad\images104.jpg | sad |
| ![images105.jpg](saved_data\test1704661267.149294\sad\images105.jpg) | saved_data\test1704661267.149294\sad\images105.jpg | sad |
| ![images106.jpg](saved_data\test1704661267.149294\sad\images106.jpg) | saved_data\test1704661267.149294\sad\images106.jpg | sad |
| ![images107.jpg](saved_data\test1704661267.149294\sad\images107.jpg) | saved_data\test1704661267.149294\sad\images107.jpg | sad |
| ![images108.jpg](saved_data\test1704661267.149294\sad\images108.jpg) | saved_data\test1704661267.149294\sad\images108.jpg | sad |
| ![images109.jpg](saved_data\test1704661267.149294\sad\images109.jpg) | saved_data\test1704661267.149294\sad\images109.jpg | sad |
| ![images11.jpg](saved_data\test1704661267.149294\sad\images11.jpg) | saved_data\test1704661267.149294\sad\images11.jpg | sad |
| ![images110.jpg](saved_data\test1704661267.149294\sad\images110.jpg) | saved_data\test1704661267.149294\sad\images110.jpg | sad |
| ![images111.jpg](saved_data\test1704661267.149294\sad\images111.jpg) | saved_data\test1704661267.149294\sad\images111.jpg | sad |
| ![images112.jpg](saved_data\test1704661267.149294\sad\images112.jpg) | saved_data\test1704661267.149294\sad\images112.jpg | sad |
| ![images113.jpg](saved_data\test1704661267.149294\sad\images113.jpg) | saved_data\test1704661267.149294\sad\images113.jpg | sad |
| ![images114.jpg](saved_data\test1704661267.149294\sad\images114.jpg) | saved_data\test1704661267.149294\sad\images114.jpg | sad |
| ![images115.jpg](saved_data\test1704661267.149294\sad\images115.jpg) | saved_data\test1704661267.149294\sad\images115.jpg | sad |
| ![images116.jpg](saved_data\test1704661267.149294\sad\images116.jpg) | saved_data\test1704661267.149294\sad\images116.jpg | sad |
| ![images117.jpg](saved_data\test1704661267.149294\sad\images117.jpg) | saved_data\test1704661267.149294\sad\images117.jpg | sad |
| ![images118.jpg](saved_data\test1704661267.149294\sad\images118.jpg) | saved_data\test1704661267.149294\sad\images118.jpg | sad |
| ![images119.jpg](saved_data\test1704661267.149294\sad\images119.jpg) | saved_data\test1704661267.149294\sad\images119.jpg | sad |
| ![images12.jpg](saved_data\test1704661267.149294\sad\images12.jpg) | saved_data\test1704661267.149294\sad\images12.jpg | sad |
| ![images120.jpg](saved_data\test1704661267.149294\sad\images120.jpg) | saved_data\test1704661267.149294\sad\images120.jpg | sad |
| ![images121.jpg](saved_data\test1704661267.149294\sad\images121.jpg) | saved_data\test1704661267.149294\sad\images121.jpg | sad |
| ![images122.jpg](saved_data\test1704661267.149294\sad\images122.jpg) | saved_data\test1704661267.149294\sad\images122.jpg | sad |
| ![images123.jpg](saved_data\test1704661267.149294\sad\images123.jpg) | saved_data\test1704661267.149294\sad\images123.jpg | sad |
| ![images124.jpg](saved_data\test1704661267.149294\sad\images124.jpg) | saved_data\test1704661267.149294\sad\images124.jpg | sad |
| ![images125.jpg](saved_data\test1704661267.149294\sad\images125.jpg) | saved_data\test1704661267.149294\sad\images125.jpg | sad |
| ![images126.jpg](saved_data\test1704661267.149294\sad\images126.jpg) | saved_data\test1704661267.149294\sad\images126.jpg | sad |
| ![images127.jpg](saved_data\test1704661267.149294\sad\images127.jpg) | saved_data\test1704661267.149294\sad\images127.jpg | sad |
| ![images128.jpg](saved_data\test1704661267.149294\sad\images128.jpg) | saved_data\test1704661267.149294\sad\images128.jpg | sad |
| ![images129.jpg](saved_data\test1704661267.149294\sad\images129.jpg) | saved_data\test1704661267.149294\sad\images129.jpg | sad |
| ![images13.jpg](saved_data\test1704661267.149294\sad\images13.jpg) | saved_data\test1704661267.149294\sad\images13.jpg | sad |
| ![images130.jpg](saved_data\test1704661267.149294\sad\images130.jpg) | saved_data\test1704661267.149294\sad\images130.jpg | sad |
| ![images131.jpg](saved_data\test1704661267.149294\sad\images131.jpg) | saved_data\test1704661267.149294\sad\images131.jpg | sad |
| ![images132.jpg](saved_data\test1704661267.149294\sad\images132.jpg) | saved_data\test1704661267.149294\sad\images132.jpg | sad |
| ![images133.jpg](saved_data\test1704661267.149294\sad\images133.jpg) | saved_data\test1704661267.149294\sad\images133.jpg | sad |
| ![images134.jpg](saved_data\test1704661267.149294\sad\images134.jpg) | saved_data\test1704661267.149294\sad\images134.jpg | sad |
| ![images135.jpg](saved_data\test1704661267.149294\sad\images135.jpg) | saved_data\test1704661267.149294\sad\images135.jpg | sad |
| ![images136.jpg](saved_data\test1704661267.149294\sad\images136.jpg) | saved_data\test1704661267.149294\sad\images136.jpg | sad |
| ![images137.jpg](saved_data\test1704661267.149294\sad\images137.jpg) | saved_data\test1704661267.149294\sad\images137.jpg | sad |
| ![images138.jpg](saved_data\test1704661267.149294\sad\images138.jpg) | saved_data\test1704661267.149294\sad\images138.jpg | sad |
| ![images139.jpg](saved_data\test1704661267.149294\sad\images139.jpg) | saved_data\test1704661267.149294\sad\images139.jpg | sad |
| ![images14.jpg](saved_data\test1704661267.149294\sad\images14.jpg) | saved_data\test1704661267.149294\sad\images14.jpg | happy |
| ![images140.jpg](saved_data\test1704661267.149294\sad\images140.jpg) | saved_data\test1704661267.149294\sad\images140.jpg | sad |
| ![images141.jpg](saved_data\test1704661267.149294\sad\images141.jpg) | saved_data\test1704661267.149294\sad\images141.jpg | sad |
| ![images142.jpg](saved_data\test1704661267.149294\sad\images142.jpg) | saved_data\test1704661267.149294\sad\images142.jpg | sad |
| ![images143.jpg](saved_data\test1704661267.149294\sad\images143.jpg) | saved_data\test1704661267.149294\sad\images143.jpg | sad |
| ![images144.jpg](saved_data\test1704661267.149294\sad\images144.jpg) | saved_data\test1704661267.149294\sad\images144.jpg | sad |
| ![images145.jpg](saved_data\test1704661267.149294\sad\images145.jpg) | saved_data\test1704661267.149294\sad\images145.jpg | sad |
| ![images146.jpg](saved_data\test1704661267.149294\sad\images146.jpg) | saved_data\test1704661267.149294\sad\images146.jpg | sad |
| ![images147.jpg](saved_data\test1704661267.149294\sad\images147.jpg) | saved_data\test1704661267.149294\sad\images147.jpg | sad |
| ![images148.jpg](saved_data\test1704661267.149294\sad\images148.jpg) | saved_data\test1704661267.149294\sad\images148.jpg | sad |
| ![images149.jpg](saved_data\test1704661267.149294\sad\images149.jpg) | saved_data\test1704661267.149294\sad\images149.jpg | sad |
| ![images15.jpg](saved_data\test1704661267.149294\sad\images15.jpg) | saved_data\test1704661267.149294\sad\images15.jpg | sad |
| ![images150.jpg](saved_data\test1704661267.149294\sad\images150.jpg) | saved_data\test1704661267.149294\sad\images150.jpg | sad |
| ![images151.jpg](saved_data\test1704661267.149294\sad\images151.jpg) | saved_data\test1704661267.149294\sad\images151.jpg | sad |
| ![images152.jpg](saved_data\test1704661267.149294\sad\images152.jpg) | saved_data\test1704661267.149294\sad\images152.jpg | sad |
| ![images153.jpg](saved_data\test1704661267.149294\sad\images153.jpg) | saved_data\test1704661267.149294\sad\images153.jpg | sad |
| ![images154.jpg](saved_data\test1704661267.149294\sad\images154.jpg) | saved_data\test1704661267.149294\sad\images154.jpg | sad |
| ![images155.jpg](saved_data\test1704661267.149294\sad\images155.jpg) | saved_data\test1704661267.149294\sad\images155.jpg | sad |
| ![images156.jpg](saved_data\test1704661267.149294\sad\images156.jpg) | saved_data\test1704661267.149294\sad\images156.jpg | sad |
| ![images157.jpg](saved_data\test1704661267.149294\sad\images157.jpg) | saved_data\test1704661267.149294\sad\images157.jpg | sad |
| ![images158.jpg](saved_data\test1704661267.149294\sad\images158.jpg) | saved_data\test1704661267.149294\sad\images158.jpg | sad |
| ![images159.jpg](saved_data\test1704661267.149294\sad\images159.jpg) | saved_data\test1704661267.149294\sad\images159.jpg | sad |
| ![images16.jpg](saved_data\test1704661267.149294\sad\images16.jpg) | saved_data\test1704661267.149294\sad\images16.jpg | sad |
| ![images160.jpg](saved_data\test1704661267.149294\sad\images160.jpg) | saved_data\test1704661267.149294\sad\images160.jpg | sad |
| ![images161.jpg](saved_data\test1704661267.149294\sad\images161.jpg) | saved_data\test1704661267.149294\sad\images161.jpg | sad |
| ![images162.jpg](saved_data\test1704661267.149294\sad\images162.jpg) | saved_data\test1704661267.149294\sad\images162.jpg | sad |
| ![images163.jpg](saved_data\test1704661267.149294\sad\images163.jpg) | saved_data\test1704661267.149294\sad\images163.jpg | sad |
| ![images164.jpg](saved_data\test1704661267.149294\sad\images164.jpg) | saved_data\test1704661267.149294\sad\images164.jpg | sad |
| ![images165.jpg](saved_data\test1704661267.149294\sad\images165.jpg) | saved_data\test1704661267.149294\sad\images165.jpg | sad |
| ![images166.jpg](saved_data\test1704661267.149294\sad\images166.jpg) | saved_data\test1704661267.149294\sad\images166.jpg | sad |
| ![images167.jpg](saved_data\test1704661267.149294\sad\images167.jpg) | saved_data\test1704661267.149294\sad\images167.jpg | sad |
| ![images168.jpg](saved_data\test1704661267.149294\sad\images168.jpg) | saved_data\test1704661267.149294\sad\images168.jpg | sad |
| ![images169.jpg](saved_data\test1704661267.149294\sad\images169.jpg) | saved_data\test1704661267.149294\sad\images169.jpg | sad |
| ![images17.jpg](saved_data\test1704661267.149294\sad\images17.jpg) | saved_data\test1704661267.149294\sad\images17.jpg | sad |
| ![images170.jpg](saved_data\test1704661267.149294\sad\images170.jpg) | saved_data\test1704661267.149294\sad\images170.jpg | sad |
| ![images171.jpg](saved_data\test1704661267.149294\sad\images171.jpg) | saved_data\test1704661267.149294\sad\images171.jpg | sad |
| ![images172.jpg](saved_data\test1704661267.149294\sad\images172.jpg) | saved_data\test1704661267.149294\sad\images172.jpg | sad |
| ![images173.jpg](saved_data\test1704661267.149294\sad\images173.jpg) | saved_data\test1704661267.149294\sad\images173.jpg | sad |
| ![images174.jpg](saved_data\test1704661267.149294\sad\images174.jpg) | saved_data\test1704661267.149294\sad\images174.jpg | sad |
| ![images175.jpg](saved_data\test1704661267.149294\sad\images175.jpg) | saved_data\test1704661267.149294\sad\images175.jpg | sad |
| ![images176.jpg](saved_data\test1704661267.149294\sad\images176.jpg) | saved_data\test1704661267.149294\sad\images176.jpg | sad |
| ![images177.jpg](saved_data\test1704661267.149294\sad\images177.jpg) | saved_data\test1704661267.149294\sad\images177.jpg | sad |
| ![images178.jpg](saved_data\test1704661267.149294\sad\images178.jpg) | saved_data\test1704661267.149294\sad\images178.jpg | sad |
| ![images179.jpg](saved_data\test1704661267.149294\sad\images179.jpg) | saved_data\test1704661267.149294\sad\images179.jpg | sad |
| ![images18.jpg](saved_data\test1704661267.149294\sad\images18.jpg) | saved_data\test1704661267.149294\sad\images18.jpg | sad |
| ![images180.jpg](saved_data\test1704661267.149294\sad\images180.jpg) | saved_data\test1704661267.149294\sad\images180.jpg | sad |
| ![images181.jpg](saved_data\test1704661267.149294\sad\images181.jpg) | saved_data\test1704661267.149294\sad\images181.jpg | sad |
| ![images182.jpg](saved_data\test1704661267.149294\sad\images182.jpg) | saved_data\test1704661267.149294\sad\images182.jpg | sad |
| ![images183.jpg](saved_data\test1704661267.149294\sad\images183.jpg) | saved_data\test1704661267.149294\sad\images183.jpg | sad |
| ![images184.jpg](saved_data\test1704661267.149294\sad\images184.jpg) | saved_data\test1704661267.149294\sad\images184.jpg | happy |
| ![images185.jpg](saved_data\test1704661267.149294\sad\images185.jpg) | saved_data\test1704661267.149294\sad\images185.jpg | sad |
| ![images186.jpg](saved_data\test1704661267.149294\sad\images186.jpg) | saved_data\test1704661267.149294\sad\images186.jpg | sad |
| ![images187.jpg](saved_data\test1704661267.149294\sad\images187.jpg) | saved_data\test1704661267.149294\sad\images187.jpg | sad |
| ![images188.jpg](saved_data\test1704661267.149294\sad\images188.jpg) | saved_data\test1704661267.149294\sad\images188.jpg | sad |
| ![images189.jpg](saved_data\test1704661267.149294\sad\images189.jpg) | saved_data\test1704661267.149294\sad\images189.jpg | sad |
| ![images19.jpg](saved_data\test1704661267.149294\sad\images19.jpg) | saved_data\test1704661267.149294\sad\images19.jpg | sad |
| ![images190.jpg](saved_data\test1704661267.149294\sad\images190.jpg) | saved_data\test1704661267.149294\sad\images190.jpg | sad |
| ![images191.jpg](saved_data\test1704661267.149294\sad\images191.jpg) | saved_data\test1704661267.149294\sad\images191.jpg | sad |
| ![images192.jpg](saved_data\test1704661267.149294\sad\images192.jpg) | saved_data\test1704661267.149294\sad\images192.jpg | sad |
| ![images193.jpg](saved_data\test1704661267.149294\sad\images193.jpg) | saved_data\test1704661267.149294\sad\images193.jpg | sad |
| ![images194.jpg](saved_data\test1704661267.149294\sad\images194.jpg) | saved_data\test1704661267.149294\sad\images194.jpg | sad |
| ![images195.jpg](saved_data\test1704661267.149294\sad\images195.jpg) | saved_data\test1704661267.149294\sad\images195.jpg | sad |
| ![images196.jpg](saved_data\test1704661267.149294\sad\images196.jpg) | saved_data\test1704661267.149294\sad\images196.jpg | sad |
| ![images197.jpg](saved_data\test1704661267.149294\sad\images197.jpg) | saved_data\test1704661267.149294\sad\images197.jpg | sad |
| ![images198.jpg](saved_data\test1704661267.149294\sad\images198.jpg) | saved_data\test1704661267.149294\sad\images198.jpg | sad |
| ![images199.jpg](saved_data\test1704661267.149294\sad\images199.jpg) | saved_data\test1704661267.149294\sad\images199.jpg | sad |
| ![images2.jpg](saved_data\test1704661267.149294\sad\images2.jpg) | saved_data\test1704661267.149294\sad\images2.jpg | sad |
| ![images20.jpg](saved_data\test1704661267.149294\sad\images20.jpg) | saved_data\test1704661267.149294\sad\images20.jpg | sad |
| ![images200.jpg](saved_data\test1704661267.149294\sad\images200.jpg) | saved_data\test1704661267.149294\sad\images200.jpg | sad |
| ![images201.jpg](saved_data\test1704661267.149294\sad\images201.jpg) | saved_data\test1704661267.149294\sad\images201.jpg | sad |
| ![images202.jpg](saved_data\test1704661267.149294\sad\images202.jpg) | saved_data\test1704661267.149294\sad\images202.jpg | sad |
| ![images203.jpg](saved_data\test1704661267.149294\sad\images203.jpg) | saved_data\test1704661267.149294\sad\images203.jpg | sad |
| ![images204.jpg](saved_data\test1704661267.149294\sad\images204.jpg) | saved_data\test1704661267.149294\sad\images204.jpg | sad |
| ![images205.jpg](saved_data\test1704661267.149294\sad\images205.jpg) | saved_data\test1704661267.149294\sad\images205.jpg | sad |
| ![images206.jpg](saved_data\test1704661267.149294\sad\images206.jpg) | saved_data\test1704661267.149294\sad\images206.jpg | sad |
| ![images21.jpg](saved_data\test1704661267.149294\sad\images21.jpg) | saved_data\test1704661267.149294\sad\images21.jpg | sad |
| ![images22.jpg](saved_data\test1704661267.149294\sad\images22.jpg) | saved_data\test1704661267.149294\sad\images22.jpg | sad |
| ![images23.jpg](saved_data\test1704661267.149294\sad\images23.jpg) | saved_data\test1704661267.149294\sad\images23.jpg | sad |
| ![images24.jpg](saved_data\test1704661267.149294\sad\images24.jpg) | saved_data\test1704661267.149294\sad\images24.jpg | sad |
| ![images25.jpg](saved_data\test1704661267.149294\sad\images25.jpg) | saved_data\test1704661267.149294\sad\images25.jpg | sad |
| ![images26.jpg](saved_data\test1704661267.149294\sad\images26.jpg) | saved_data\test1704661267.149294\sad\images26.jpg | sad |
| ![images27.jpg](saved_data\test1704661267.149294\sad\images27.jpg) | saved_data\test1704661267.149294\sad\images27.jpg | sad |
| ![images28.jpg](saved_data\test1704661267.149294\sad\images28.jpg) | saved_data\test1704661267.149294\sad\images28.jpg | sad |
| ![images29.jpg](saved_data\test1704661267.149294\sad\images29.jpg) | saved_data\test1704661267.149294\sad\images29.jpg | sad |
| ![images3.jpg](saved_data\test1704661267.149294\sad\images3.jpg) | saved_data\test1704661267.149294\sad\images3.jpg | sad |
| ![images30.jpg](saved_data\test1704661267.149294\sad\images30.jpg) | saved_data\test1704661267.149294\sad\images30.jpg | sad |
| ![images31.jpg](saved_data\test1704661267.149294\sad\images31.jpg) | saved_data\test1704661267.149294\sad\images31.jpg | sad |
| ![images32.jpg](saved_data\test1704661267.149294\sad\images32.jpg) | saved_data\test1704661267.149294\sad\images32.jpg | sad |
| ![images33.jpg](saved_data\test1704661267.149294\sad\images33.jpg) | saved_data\test1704661267.149294\sad\images33.jpg | sad |
| ![images34.jpg](saved_data\test1704661267.149294\sad\images34.jpg) | saved_data\test1704661267.149294\sad\images34.jpg | sad |
| ![images35.jpg](saved_data\test1704661267.149294\sad\images35.jpg) | saved_data\test1704661267.149294\sad\images35.jpg | sad |
| ![images36.jpg](saved_data\test1704661267.149294\sad\images36.jpg) | saved_data\test1704661267.149294\sad\images36.jpg | sad |
| ![images37.jpg](saved_data\test1704661267.149294\sad\images37.jpg) | saved_data\test1704661267.149294\sad\images37.jpg | sad |
| ![images38.jpg](saved_data\test1704661267.149294\sad\images38.jpg) | saved_data\test1704661267.149294\sad\images38.jpg | sad |
| ![images39.jpg](saved_data\test1704661267.149294\sad\images39.jpg) | saved_data\test1704661267.149294\sad\images39.jpg | sad |
| ![images4.jpg](saved_data\test1704661267.149294\sad\images4.jpg) | saved_data\test1704661267.149294\sad\images4.jpg | sad |
| ![images40.jpg](saved_data\test1704661267.149294\sad\images40.jpg) | saved_data\test1704661267.149294\sad\images40.jpg | sad |
| ![images41.jpg](saved_data\test1704661267.149294\sad\images41.jpg) | saved_data\test1704661267.149294\sad\images41.jpg | sad |
| ![images42.jpg](saved_data\test1704661267.149294\sad\images42.jpg) | saved_data\test1704661267.149294\sad\images42.jpg | sad |
| ![images43.jpg](saved_data\test1704661267.149294\sad\images43.jpg) | saved_data\test1704661267.149294\sad\images43.jpg | sad |
| ![images44.jpg](saved_data\test1704661267.149294\sad\images44.jpg) | saved_data\test1704661267.149294\sad\images44.jpg | sad |
| ![images45.jpg](saved_data\test1704661267.149294\sad\images45.jpg) | saved_data\test1704661267.149294\sad\images45.jpg | sad |
| ![images46.jpg](saved_data\test1704661267.149294\sad\images46.jpg) | saved_data\test1704661267.149294\sad\images46.jpg | sad |
| ![images47.jpg](saved_data\test1704661267.149294\sad\images47.jpg) | saved_data\test1704661267.149294\sad\images47.jpg | sad |
| ![images48.jpg](saved_data\test1704661267.149294\sad\images48.jpg) | saved_data\test1704661267.149294\sad\images48.jpg | sad |
| ![images49.jpg](saved_data\test1704661267.149294\sad\images49.jpg) | saved_data\test1704661267.149294\sad\images49.jpg | sad |
| ![images5.jpg](saved_data\test1704661267.149294\sad\images5.jpg) | saved_data\test1704661267.149294\sad\images5.jpg | sad |
| ![images50.jpg](saved_data\test1704661267.149294\sad\images50.jpg) | saved_data\test1704661267.149294\sad\images50.jpg | sad |
| ![images51.jpg](saved_data\test1704661267.149294\sad\images51.jpg) | saved_data\test1704661267.149294\sad\images51.jpg | happy |
| ![images52.jpg](saved_data\test1704661267.149294\sad\images52.jpg) | saved_data\test1704661267.149294\sad\images52.jpg | sad |
| ![images53.jpg](saved_data\test1704661267.149294\sad\images53.jpg) | saved_data\test1704661267.149294\sad\images53.jpg | sad |
| ![images54.jpg](saved_data\test1704661267.149294\sad\images54.jpg) | saved_data\test1704661267.149294\sad\images54.jpg | sad |
| ![images55.jpg](saved_data\test1704661267.149294\sad\images55.jpg) | saved_data\test1704661267.149294\sad\images55.jpg | sad |
| ![images56.jpg](saved_data\test1704661267.149294\sad\images56.jpg) | saved_data\test1704661267.149294\sad\images56.jpg | sad |
| ![images57.jpg](saved_data\test1704661267.149294\sad\images57.jpg) | saved_data\test1704661267.149294\sad\images57.jpg | sad |
| ![images58.jpg](saved_data\test1704661267.149294\sad\images58.jpg) | saved_data\test1704661267.149294\sad\images58.jpg | sad |
| ![images59.jpg](saved_data\test1704661267.149294\sad\images59.jpg) | saved_data\test1704661267.149294\sad\images59.jpg | sad |
| ![images6.jpg](saved_data\test1704661267.149294\sad\images6.jpg) | saved_data\test1704661267.149294\sad\images6.jpg | sad |
| ![images60.jpg](saved_data\test1704661267.149294\sad\images60.jpg) | saved_data\test1704661267.149294\sad\images60.jpg | sad |
| ![images61.jpg](saved_data\test1704661267.149294\sad\images61.jpg) | saved_data\test1704661267.149294\sad\images61.jpg | sad |
| ![images62.jpg](saved_data\test1704661267.149294\sad\images62.jpg) | saved_data\test1704661267.149294\sad\images62.jpg | sad |
| ![images63.jpg](saved_data\test1704661267.149294\sad\images63.jpg) | saved_data\test1704661267.149294\sad\images63.jpg | sad |
| ![images64.jpg](saved_data\test1704661267.149294\sad\images64.jpg) | saved_data\test1704661267.149294\sad\images64.jpg | sad |
| ![images65.jpg](saved_data\test1704661267.149294\sad\images65.jpg) | saved_data\test1704661267.149294\sad\images65.jpg | sad |
| ![images66.jpg](saved_data\test1704661267.149294\sad\images66.jpg) | saved_data\test1704661267.149294\sad\images66.jpg | sad |
| ![images67.jpg](saved_data\test1704661267.149294\sad\images67.jpg) | saved_data\test1704661267.149294\sad\images67.jpg | sad |
| ![images68.jpg](saved_data\test1704661267.149294\sad\images68.jpg) | saved_data\test1704661267.149294\sad\images68.jpg | sad |
| ![images69.jpg](saved_data\test1704661267.149294\sad\images69.jpg) | saved_data\test1704661267.149294\sad\images69.jpg | sad |
| ![images7.jpg](saved_data\test1704661267.149294\sad\images7.jpg) | saved_data\test1704661267.149294\sad\images7.jpg | sad |
| ![images70.jpg](saved_data\test1704661267.149294\sad\images70.jpg) | saved_data\test1704661267.149294\sad\images70.jpg | sad |
| ![images71.jpg](saved_data\test1704661267.149294\sad\images71.jpg) | saved_data\test1704661267.149294\sad\images71.jpg | sad |
| ![images72.jpg](saved_data\test1704661267.149294\sad\images72.jpg) | saved_data\test1704661267.149294\sad\images72.jpg | sad |
| ![images73.jpg](saved_data\test1704661267.149294\sad\images73.jpg) | saved_data\test1704661267.149294\sad\images73.jpg | sad |
| ![images74.jpg](saved_data\test1704661267.149294\sad\images74.jpg) | saved_data\test1704661267.149294\sad\images74.jpg | sad |
| ![images75.jpg](saved_data\test1704661267.149294\sad\images75.jpg) | saved_data\test1704661267.149294\sad\images75.jpg | happy |
| ![images76.jpg](saved_data\test1704661267.149294\sad\images76.jpg) | saved_data\test1704661267.149294\sad\images76.jpg | sad |
| ![images77.jpg](saved_data\test1704661267.149294\sad\images77.jpg) | saved_data\test1704661267.149294\sad\images77.jpg | sad |
| ![images78.jpg](saved_data\test1704661267.149294\sad\images78.jpg) | saved_data\test1704661267.149294\sad\images78.jpg | sad |
| ![images79.jpg](saved_data\test1704661267.149294\sad\images79.jpg) | saved_data\test1704661267.149294\sad\images79.jpg | sad |
| ![images8.jpg](saved_data\test1704661267.149294\sad\images8.jpg) | saved_data\test1704661267.149294\sad\images8.jpg | sad |
| ![images80.jpg](saved_data\test1704661267.149294\sad\images80.jpg) | saved_data\test1704661267.149294\sad\images80.jpg | sad |
| ![images81.jpg](saved_data\test1704661267.149294\sad\images81.jpg) | saved_data\test1704661267.149294\sad\images81.jpg | happy |
| ![images82.jpg](saved_data\test1704661267.149294\sad\images82.jpg) | saved_data\test1704661267.149294\sad\images82.jpg | sad |
| ![images83.jpg](saved_data\test1704661267.149294\sad\images83.jpg) | saved_data\test1704661267.149294\sad\images83.jpg | sad |
| ![images84.jpg](saved_data\test1704661267.149294\sad\images84.jpg) | saved_data\test1704661267.149294\sad\images84.jpg | sad |
| ![images85.jpg](saved_data\test1704661267.149294\sad\images85.jpg) | saved_data\test1704661267.149294\sad\images85.jpg | sad |
| ![images86.jpg](saved_data\test1704661267.149294\sad\images86.jpg) | saved_data\test1704661267.149294\sad\images86.jpg | sad |
| ![images87.jpg](saved_data\test1704661267.149294\sad\images87.jpg) | saved_data\test1704661267.149294\sad\images87.jpg | sad |
| ![images88.jpg](saved_data\test1704661267.149294\sad\images88.jpg) | saved_data\test1704661267.149294\sad\images88.jpg | sad |
| ![images89.jpg](saved_data\test1704661267.149294\sad\images89.jpg) | saved_data\test1704661267.149294\sad\images89.jpg | sad |
| ![images9.jpg](saved_data\test1704661267.149294\sad\images9.jpg) | saved_data\test1704661267.149294\sad\images9.jpg | sad |
| ![images90.jpg](saved_data\test1704661267.149294\sad\images90.jpg) | saved_data\test1704661267.149294\sad\images90.jpg | happy |
| ![images91.jpg](saved_data\test1704661267.149294\sad\images91.jpg) | saved_data\test1704661267.149294\sad\images91.jpg | sad |
| ![images92.jpg](saved_data\test1704661267.149294\sad\images92.jpg) | saved_data\test1704661267.149294\sad\images92.jpg | sad |
| ![images93.jpg](saved_data\test1704661267.149294\sad\images93.jpg) | saved_data\test1704661267.149294\sad\images93.jpg | sad |
| ![images94.jpg](saved_data\test1704661267.149294\sad\images94.jpg) | saved_data\test1704661267.149294\sad\images94.jpg | sad |
| ![images95.jpg](saved_data\test1704661267.149294\sad\images95.jpg) | saved_data\test1704661267.149294\sad\images95.jpg | sad |
| ![images96.jpg](saved_data\test1704661267.149294\sad\images96.jpg) | saved_data\test1704661267.149294\sad\images96.jpg | sad |
| ![images97.jpg](saved_data\test1704661267.149294\sad\images97.jpg) | saved_data\test1704661267.149294\sad\images97.jpg | sad |
| ![images98.jpg](saved_data\test1704661267.149294\sad\images98.jpg) | saved_data\test1704661267.149294\sad\images98.jpg | sad |
| ![images99.jpg](saved_data\test1704661267.149294\sad\images99.jpg) | saved_data\test1704661267.149294\sad\images99.jpg | sad |
| ![iStock_000001932580XSmall.jpg](saved_data\test1704661267.149294\sad\iStock_000001932580XSmall.jpg) | saved_data\test1704661267.149294\sad\iStock_000001932580XSmall.jpg | sad |
| ![jack-lucas-smith-Zxq0dvmRyIo-unsplash-1024x701.jpg](saved_data\test1704661267.149294\sad\jack-lucas-smith-Zxq0dvmRyIo-unsplash-1024x701.jpg) | saved_data\test1704661267.149294\sad\jack-lucas-smith-Zxq0dvmRyIo-unsplash-1024x701.jpg | sad |
| ![l-person-disappointed-of-corporate-job-fail-or-mistake-in-studio-fit_400_400.jpg](saved_data\test1704661267.149294\sad\l-person-disappointed-of-corporate-job-fail-or-mistake-in-studio-fit_400_400.jpg) | saved_data\test1704661267.149294\sad\l-person-disappointed-of-corporate-job-fail-or-mistake-in-studio-fit_400_400.jpg | sad |
| ![lifetime-member-tee-01-1000x1000.jpg](saved_data\test1704661267.149294\sad\lifetime-member-tee-01-1000x1000.jpg) | saved_data\test1704661267.149294\sad\lifetime-member-tee-01-1000x1000.jpg | sad |
| ![lonely-depressed-person-sitting-near-brick-wall_181624-30778.jpg](saved_data\test1704661267.149294\sad\lonely-depressed-person-sitting-near-brick-wall_181624-30778.jpg) | saved_data\test1704661267.149294\sad\lonely-depressed-person-sitting-near-brick-wall_181624-30778.jpg | sad |
| ![lonely-young-woman-in-the-rain-feeling-sadness-and-hopelessness-generated-by-ai-photo.jpg](saved_data\test1704661267.149294\sad\lonely-young-woman-in-the-rain-feeling-sadness-and-hopelessness-generated-by-ai-photo.jpg) | saved_data\test1704661267.149294\sad\lonely-young-woman-in-the-rain-feeling-sadness-and-hopelessness-generated-by-ai-photo.jpg | sad |
| ![lovepik-a-sad-old-man-png-image_401101920_wh1200.png](saved_data\test1704661267.149294\sad\lovepik-a-sad-old-man-png-image_401101920_wh1200.png) | saved_data\test1704661267.149294\sad\lovepik-a-sad-old-man-png-image_401101920_wh1200.png | sad |
| ![man-sat-bent-his-knees-holding-his-hands-face-base-tree-there-is-water-around_1150-16341.jpg](saved_data\test1704661267.149294\sad\man-sat-bent-his-knees-holding-his-hands-face-base-tree-there-is-water-around_1150-16341.jpg) | saved_data\test1704661267.149294\sad\man-sat-bent-his-knees-holding-his-hands-face-base-tree-there-is-water-around_1150-16341.jpg | sad |
| ![man-tears-tear-look.jpg](saved_data\test1704661267.149294\sad\man-tears-tear-look.jpg) | saved_data\test1704661267.149294\sad\man-tears-tear-look.jpg | sad |
| ![man-with-head-down-300x300.jpg](saved_data\test1704661267.149294\sad\man-with-head-down-300x300.jpg) | saved_data\test1704661267.149294\sad\man-with-head-down-300x300.jpg | sad |
| ![n-rendering-frustrated-upset-man-sitting-white-people-man-character-53250684.jpg](saved_data\test1704661267.149294\sad\n-rendering-frustrated-upset-man-sitting-white-people-man-character-53250684.jpg) | saved_data\test1704661267.149294\sad\n-rendering-frustrated-upset-man-sitting-white-people-man-character-53250684.jpg | sad |
| ![nal-man-digital-illustration-transparent-background-person-sitting-thumbnail.png](saved_data\test1704661267.149294\sad\nal-man-digital-illustration-transparent-background-person-sitting-thumbnail.png) | saved_data\test1704661267.149294\sad\nal-man-digital-illustration-transparent-background-person-sitting-thumbnail.png | sad |
| ![people-2567395_640.jpg](saved_data\test1704661267.149294\sad\people-2567395_640.jpg) | saved_data\test1704661267.149294\sad\people-2567395_640.jpg | sad |
| ![people-lady-sadness-about-love-from-boyfriend-she-feeling-broken-heart-photo.jpg](saved_data\test1704661267.149294\sad\people-lady-sadness-about-love-from-boyfriend-she-feeling-broken-heart-photo.jpg) | saved_data\test1704661267.149294\sad\people-lady-sadness-about-love-from-boyfriend-she-feeling-broken-heart-photo.jpg | sad |
| ![people-person-sad-uninspired-down-depressed-512.png](saved_data\test1704661267.149294\sad\people-person-sad-uninspired-down-depressed-512.png) | saved_data\test1704661267.149294\sad\people-person-sad-uninspired-down-depressed-512.png | sad |
| ![person-super-depressed.jpg](saved_data\test1704661267.149294\sad\person-super-depressed.jpg) | saved_data\test1704661267.149294\sad\person-super-depressed.jpg | sad |
| ![png-transparent-sadness-cartoon-man-cartoon-sad-people-angle-white-face.png](saved_data\test1704661267.149294\sad\png-transparent-sadness-cartoon-man-cartoon-sad-people-angle-white-face.png) | saved_data\test1704661267.149294\sad\png-transparent-sadness-cartoon-man-cartoon-sad-people-angle-white-face.png | sad |
| ![pngtree-woman-looking-sad-in-black-and-white-picture-image_2770858.jpg](saved_data\test1704661267.149294\sad\pngtree-woman-looking-sad-in-black-and-white-picture-image_2770858.jpg) | saved_data\test1704661267.149294\sad\pngtree-woman-looking-sad-in-black-and-white-picture-image_2770858.jpg | sad |
| ![pngtree-woman-looking-sad-in-the-rain-picture-image_2771069.jpg](saved_data\test1704661267.149294\sad\pngtree-woman-looking-sad-in-the-rain-picture-image_2771069.jpg) | saved_data\test1704661267.149294\sad\pngtree-woman-looking-sad-in-the-rain-picture-image_2771069.jpg | sad |
| ![sad-depressed-man.jpg](saved_data\test1704661267.149294\sad\sad-depressed-man.jpg) | saved_data\test1704661267.149294\sad\sad-depressed-man.jpg | sad |
| ![sad-glance-mm-nisan-kandilcioglu.jpg](saved_data\test1704661267.149294\sad\sad-glance-mm-nisan-kandilcioglu.jpg) | saved_data\test1704661267.149294\sad\sad-glance-mm-nisan-kandilcioglu.jpg | sad |
| ![sad-group-people-problems-17033671.jpg](saved_data\test1704661267.149294\sad\sad-group-people-problems-17033671.jpg) | saved_data\test1704661267.149294\sad\sad-group-people-problems-17033671.jpg | sad |
| ![Sad-man-being-consoled-by-friends-in-group-therapy.jpg](saved_data\test1704661267.149294\sad\Sad-man-being-consoled-by-friends-in-group-therapy.jpg) | saved_data\test1704661267.149294\sad\Sad-man-being-consoled-by-friends-in-group-therapy.jpg | sad |
| ![Sad-people-Icon-Graphics-4929378-1.jpg](saved_data\test1704661267.149294\sad\Sad-people-Icon-Graphics-4929378-1.jpg) | saved_data\test1704661267.149294\sad\Sad-people-Icon-Graphics-4929378-1.jpg | sad |
| ![sad-people-vector-26812552.jpg](saved_data\test1704661267.149294\sad\sad-people-vector-26812552.jpg) | saved_data\test1704661267.149294\sad\sad-people-vector-26812552.jpg | sad |
| ![sad-people-worried-people-thinking-depression-alone-shoulder-sitting-joint-png-clipart-thumbnail.jpg](saved_data\test1704661267.149294\sad\sad-people-worried-people-thinking-depression-alone-shoulder-sitting-joint-png-clipart-thumbnail.jpg) | saved_data\test1704661267.149294\sad\sad-people-worried-people-thinking-depression-alone-shoulder-sitting-joint-png-clipart-thumbnail.jpg | sad |
| ![Sad-People.jpg](saved_data\test1704661267.149294\sad\Sad-People.jpg) | saved_data\test1704661267.149294\sad\Sad-People.jpg | sad |
| ![sad-person-concept-vector-26538685.jpg](saved_data\test1704661267.149294\sad\sad-person-concept-vector-26538685.jpg) | saved_data\test1704661267.149294\sad\sad-person-concept-vector-26538685.jpg | sad |
| ![sad.jpg](saved_data\test1704661267.149294\sad\sad.jpg) | saved_data\test1704661267.149294\sad\sad.jpg | sad |
| ![sad2.jpg](saved_data\test1704661267.149294\sad\sad2.jpg) | saved_data\test1704661267.149294\sad\sad2.jpg | sad |
| ![sadness.jpg](saved_data\test1704661267.149294\sad\sadness.jpg) | saved_data\test1704661267.149294\sad\sadness.jpg | sad |
| ![sadpeople.jpg](saved_data\test1704661267.149294\sad\sadpeople.jpg) | saved_data\test1704661267.149294\sad\sadpeople.jpg | sad |
| ![sadpersonas-risks-symptoms-suicide.jpg](saved_data\test1704661267.149294\sad\sadpersonas-risks-symptoms-suicide.jpg) | saved_data\test1704661267.149294\sad\sadpersonas-risks-symptoms-suicide.jpg | sad |
| ![thinking-people-sad-people-worried-people-depression-alone-clothing-jeans-pants-png-clipart.jpg](saved_data\test1704661267.149294\sad\thinking-people-sad-people-worried-people-depression-alone-clothing-jeans-pants-png-clipart.jpg) | saved_data\test1704661267.149294\sad\thinking-people-sad-people-worried-people-depression-alone-clothing-jeans-pants-png-clipart.jpg | sad |
| ![_2539df08-4f50-11e6-85e3-522dd231fa74.jpg](saved_data\test1704661267.149294\sad\_2539df08-4f50-11e6-85e3-522dd231fa74.jpg) | saved_data\test1704661267.149294\sad\_2539df08-4f50-11e6-85e3-522dd231fa74.jpg | sad |

Number of correct sad people predictions 304 from a total number of 312 images.
