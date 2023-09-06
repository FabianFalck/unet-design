
import os
import time
import numpy as np
import warnings
import scipy
import SimpleITK as sitk
import argparse
# import tensorflow as tf
# from keras.models import Model
# from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation
# from keras.layers.merge import concatenate
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint
# from keras import backend as K
# from keras.preprocessing.image import apply_transform, transform_matrix_offset_center
from evaluation import getDSC, getHausdorff, getLesionDetection, getAVD, getImages_preprocess
# K.set_image_data_format('channels_last')

from test_leave_one_out import Utrecht_preprocessing, GE3T_preprocessing

from hyperparams import Hyperparams

rows_standard = 200
cols_standard = 200
thresh_FLAIR = 70      #to mask the brain
thresh_T1 = 30
smooth=1.


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Multi-resolution diffusion hyperparameters.')

    parser.add_argument('--what_to_preprocess', type=str, default='train', metavar='N')

    args = parser.parse_args()
    H = Hyperparams(args.__dict__)
    

    # select by user
    what_to_preprocess = H.what_to_preprocess  # 'train' OR 'test'

    if what_to_preprocess == 'train':
        train_data_path = 'data/training'
        save_suffix = '_train'
        n_patients_0 = 20
        n_patients_1 = 20
        n_patients_2 = 20
    elif what_to_preprocess == 'test':
        train_data_path = 'data/test'
        save_suffix = '_test'
        n_patients_0 = 30
        n_patients_1 = 30
        n_patients_2 = 30

    print("Preprocessing started!")

    # arguments of function
    flair=True
    t1=True
    full=True
    first5=True
    aug=True
    verbose=False

    imgs_test_list, testImage_list = [], []

    n_total_patients = n_patients_0 + n_patients_1 + n_patients_2
    n_patients_0_1 = n_patients_0 + n_patients_1

    for patient in range(n_total_patients):
        if patient < n_patients_0: dir = os.path.join(train_data_path, 'Utrecht/')
        elif patient < n_patients_0_1: dir = os.path.join(train_data_path, 'Singapore/')
        else: dir = os.path.join(train_data_path, 'Amsterdam/GE3T/')
        dirs = os.listdir(dir)
        dirs.sort()
        dir += dirs[patient % n_patients_0]
        # Fluid-attenuated inversion recovery (FLAIR) is an advanced magnetic resonance imaging sequence 
        # that reveals tissue T2 prolongation with cerebrospinal fluid suppression, allowing detection of superficial brain lesions.
        FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')  # an image modality
        # T1 (longitudinal relaxation time) is the time constant which determines the rate at which excited protons return to equilibrium. 
        # It is a measure of the time taken for spinning protons to realign with the external magnetic field.
        T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')  # another image modality
        FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
        T1_array = sitk.GetArrayFromImage(T1_image)
        
        # if patient >= 40: 
        #     print("stop")

        if patient < n_patients_0_1: imgs_test = Utrecht_preprocessing(FLAIR_array, T1_array)
        else: imgs_test = GE3T_preprocessing(FLAIR_array, T1_array)
        if not flair: imgs_test = imgs_test[..., 1:2].copy()
        if not t1: imgs_test = imgs_test[..., 0:1].copy()

        # img_shape is (48, 200, 200, 2)
        # 48: slices
        # 200, 200: probably spatial dimensions
        # 2: flair and t1
        img_shape = (rows_standard, cols_standard, flair+t1)  

        # model = get_unet(img_shape, first5)
        # model_path = 'models/'
        #if you want to test three models ensemble, just do like this: pred = (pred_1+pred_2+pred_3)/3
        # model.load_weights(model_path + str(patient) + '.h5')
        # pred = model.predict(imgs_test, batch_size=1, verbose=verbose)
        # pred[pred > 0.5] = 1.
        # pred[pred <= 0.5] = 0.
        # if patient < 40: original_pred = Utrecht_postprocessing(FLAIR_array, pred)
        # else: original_pred = GE3T_postprocessing(FLAIR_array, pred)
        # filename_resultImage = model_path + str(patient) + '.nii.gz'
        # sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )

        # testImage contains MASKS!!!!!!!
        # see Manual reference standard in documentation --> contains 3 labels, the first two being background (negative) vs. WMH (positive) 
        filename_testImage = os.path.join(dir + '/wmh.nii.gz')  
        testImage = getImages_preprocess(filename_testImage)  #   , resultImage   ;;;   , filename_resultImage

        # convert to numpy 
        testImage = sitk.GetArrayFromImage(testImage)  # shape: (48, 240, 240)  -> binary mask

        # correct shape
        image_rows_Dataset = testImage.shape[1]
        image_cols_Dataset = testImage.shape[2]
        if patient < n_patients_0_1: 
            testImage = testImage[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
        else: 
            channel_num = 2
            start_cut = 46
            num_selected_slice = testImage.shape[0]

            testImage_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
            testImage_suitable[...] = np.min(testImage)
            testImage_suitable[:, :, int(cols_standard/2-image_cols_Dataset/2):int(cols_standard/2+image_cols_Dataset/2)] = testImage[:, start_cut:(start_cut+rows_standard), :]
            testImage = testImage_suitable

        # dsc = getDSC(testImage, resultImage)
        # avd = getAVD(testImage, resultImage) 
        # h95 = getHausdorff(testImage, resultImage)
        # recall, f1 = getLesionDetection(testImage, resultImage)
        # return dsc, h95, avd, recall, f1

        imgs_test_list.append(imgs_test)
        testImage_list.append(testImage)

        print('Patient: ', patient)
        print("imgs_test: ", imgs_test.shape, "testImage: ", testImage.shape)

    # concatenate along the 0th axis
    imgs_test = np.concatenate(imgs_test_list, axis=0)
    testImage = np.concatenate(testImage_list, axis=0)

    # save the data
    print("saving the preprocessed data...")
    np.save('data_preprocessed/images_three_datasets_sorted' + save_suffix + '.npy', imgs_test)
    np.save('data_preprocessed/masks_three_datasets_sorted' + save_suffix + '.npy', testImage)


    print("Preprocessing finished!")