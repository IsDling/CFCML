import SimpleITK as sitk
import numpy as np
import os
from scipy.ndimage import zoom
from scipy import ndimage
import random
import datetime
import torch
import math
from torchvision import transforms


def Standardize(images):
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    Mean and std parameter have to be provided explicitly.
    new: z-score is used but keep the background with zero!
    """
    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)
    mask_location = images.sum(0) > 0
    # print(mask_location.shape)
    for k in range(images.shape[0]):
        image = images[k,...]
        image = np.array(image, dtype='float32')
        mask_area = image[mask_location]
        image[mask_location] -= mask_area.mean()
        image[mask_location] /= mask_area.std()
        images[k,...] = image
    # return (image - self.mean) / np.clip(self.std, a_min=self.eps, a_max=None)
    # print(images.mean(),images.std())
    # print(images.shape)
    return images

def multimodel_Standard(image):
    for i in range(image.shape[0]):
        img = image[i, ...]
        new_img = Standardize(img)
        if i == 0:
            out_image = new_img
        else:
            out_image = np.concatenate((out_image, new_img), axis=0)
    out_image = out_image.copy()
    return out_image

def flip(all_img, flip_type):
    # 0为前后翻转, 1为上下翻转, 2为左右翻转
    return np.flip(all_img, flip_type)

def gaussian_noise(img,ifonline=0,scale=10):
    if ifonline:
        scale = random.randint(2,10)
    noise = np.random.normal(np.mean(img), np.std(img), img.shape)
    return img + noise/scale

def save_operation(out_all_img, new_patient_path, da_type, i):
    slice_num = int(out_all_img.shape[0] / 3)
    out_t1_img = out_all_img[:slice_num]
    out_t2_img = out_all_img[slice_num:slice_num * 2]
    out_adc_img = out_all_img[slice_num * 2:]

    out_t1_img = sitk.GetImageFromArray(out_t1_img)
    out_t2_img = sitk.GetImageFromArray(out_t2_img)
    out_adc_img = sitk.GetImageFromArray(out_adc_img)

    sitk.WriteImage(out_t1_img, os.path.join(new_patient_path, 't1_bbox_{}{}.nii.gz'.format(da_type,str(i))))
    sitk.WriteImage(out_t2_img, os.path.join(new_patient_path, 't2_bbox_{}{}.nii.gz'.format(da_type,str(i))))
    sitk.WriteImage(out_adc_img, os.path.join(new_patient_path, 'adc_bbox_{}{}.nii.gz'.format(da_type,str(i))))

def noise_operation(all_img, new_patient_path=None, ifonline=0):

    slice_num = int(all_img.shape[0] / 3)
    out_t1_img = gaussian_noise(all_img[:slice_num], ifonline=ifonline)
    out_t2_img = gaussian_noise(all_img[slice_num:slice_num * 2], ifonline=ifonline)
    out_adc_img = gaussian_noise(all_img[slice_num * 2:], ifonline=ifonline)
    out_all_img = np.concatenate((out_t1_img, out_t2_img, out_adc_img), axis=0)

    if ifonline:
        return out_all_img
    else:
        save_operation(out_all_img, new_patient_path, 'noise', i=0)

def flip_operation(all_img, new_patient_path=None, flip_type=None, ifonline=0):
    if flip_type == 0:
        slice_num = int(all_img.shape[0] / 3)
        out_t1_img = flip(all_img[:slice_num], flip_type)
        out_t2_img = flip(all_img[slice_num:slice_num * 2], flip_type)
        out_adc_img = flip(all_img[slice_num * 2:], flip_type)
        out_all_img = np.concatenate((out_t1_img,out_t2_img,out_adc_img),axis=0)
    else:
        out_all_img = flip(all_img, flip_type)

    if ifonline:
        return out_all_img
    else:
        save_operation(out_all_img, new_patient_path, 'flip', flip_type)

def random_crop_operation(all_img, new_patient_path=None, i=None, ifonline=0, image_size=128):
    all_img = torch.from_numpy(all_img.copy()) # 参考 https://cloud.tencent.com/developer/article/2068802
    out_all_img = transforms.RandomResizedCrop((image_size, image_size), scale=(0.75, 1), antialias=True)(all_img)

    if ifonline:
        return np.array(out_all_img)
    else:
        save_operation(np.array(out_all_img), new_patient_path, 'crop', i)

def random_erasing_operation(all_img, new_patient_path=None, i=None, ifonline=0):
    all_img = torch.from_numpy(all_img.copy())
    out_all_img = transforms.RandomErasing(p=1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=(0), inplace=False)(all_img)

    if ifonline:
        return np.array(out_all_img)
    else:
        save_operation(np.array(out_all_img), new_patient_path, 'erasing', i)

def online_aug(image_array, image_size=128, online_aug_type=None):
    # image_array [24*3, 128, 128]
    ifrandom_crop = 1
    ifflip = 1
    ifnoise = 1
    iferasing = 1

    if online_aug_type =='random_crop':
        ifrandom_crop = random.random()
    elif online_aug_type =='flip':
        ifflip = random.random()
    elif online_aug_type =='noise':
        ifnoise = random.random()
    elif online_aug_type =='erasing':
        iferasing = random.random()
    else:
        ifrandom_crop = random.random()
        ifflip = random.random()
        ifnoise = random.random()
        iferasing = random.random()

    out_image = image_array
    if ifrandom_crop < 0.3:
        out_image = random_crop_operation(out_image,ifonline=1,image_size=image_size)
    if ifflip < 0.3:
        flip_type = random.randint(0,2)
        out_image = flip_operation(out_image,flip_type=flip_type,ifonline=1)
    if ifnoise < 0.3:
        out_image = noise_operation(out_image,ifonline=1)
    if iferasing < 0.3:
        out_image = random_erasing_operation(out_image, ifonline=1)

    return out_image

def aug_BBox(path, image_size):
    for fold_path_name in os.listdir(path):
        fold_path = os.path.join(path, fold_path_name)
        for traintest_path_name in os.listdir(fold_path):
            traintest_path = os.path.join(fold_path, traintest_path_name)
            # 仅对训练集数据进行数据增强
            if traintest_path_name == 'train':
                for grade_path_name in os.listdir(traintest_path):
                    grade_path = os.path.join(traintest_path, grade_path_name)
                    for patient_path_name in os.listdir(grade_path):
                        patient_path = os.path.join(grade_path, patient_path_name)
                        print('deal with ', patient_path)

                        new_patient_path = patient_path

                        t1_img = sitk.ReadImage(os.path.join(patient_path, 't1_bbox.nii.gz'))
                        t1_img = sitk.GetArrayFromImage(t1_img)

                        t2_img = sitk.ReadImage(os.path.join(patient_path, 't2_bbox.nii.gz'))
                        t2_img = sitk.GetArrayFromImage(t2_img)

                        adc_img = sitk.ReadImage(os.path.join(patient_path, 'adc_bbox.nii.gz'))
                        adc_img = sitk.GetArrayFromImage(adc_img)

                        all_img = np.concatenate((t1_img, t2_img, adc_img), axis=0)


                        if grade_path_name == 'Grade_1':  # 2倍增强, random_erasing
                            for i in range(1):
                                random_erasing_operation(all_img, new_patient_path, i)
                        elif grade_path_name == 'Grade_2_invasion':  # 21倍增强
                            # noise
                            noise_operation(all_img, new_patient_path)
                            # flip, 0为前后翻转, 1为上下翻转, 2为左右翻转
                            for i in range(3):
                                flip_operation(all_img, new_patient_path, i)
                            # random_erasing
                            for i in range(5):
                                random_erasing_operation(all_img, new_patient_path, i)
                            # random_crop
                            for i in range(11):
                                random_crop_operation(all_img, new_patient_path, i, image_size)
                        elif grade_path_name == 'Grade_2_noninvasion':  # 15倍增强
                            # noise
                            noise_operation(all_img, new_patient_path)
                            # flip
                            for i in range(3):
                                flip_operation(all_img, new_patient_path, i)
                            # random_erasing
                            for i in range(4):
                                random_erasing_operation(all_img, new_patient_path, i)
                            # random_crop
                            for i in range(5):
                                random_crop_operation(all_img, new_patient_path, i, image_size)
                        else:
                            pass

def get_image(path):
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)
    return img

def calc_online_aug_time():
    t1_img = get_image('/media/ExtHDD02/ltl/Data/MEN/cv3folders_multicls/1fold/test/Grade_1/PN0002735-DONG_XIU_MEI/t1_bbox.nii.gz')
    t2_img = get_image('/media/ExtHDD02/ltl/Data/MEN/cv3folders_multicls/1fold/test/Grade_1/PN0002735-DONG_XIU_MEI/t2_bbox.nii.gz')
    adc_img = get_image('/media/ExtHDD02/ltl/Data/MEN/cv3folders_multicls/1fold/test/Grade_1/PN0002735-DONG_XIU_MEI/adc_bbox.nii.gz')

    img_out = np.concatenate((t1_img, t2_img, adc_img), axis=0)
    all_img = torch.from_numpy(img_out)

    start_time = datetime.datetime.now()
    out_image = random_crop_operation(img_out, ifonline=1)
    end_time = datetime.datetime.now()
    print('random_crop:', end_time - start_time)

    start_time = datetime.datetime.now()
    out_image = flip_operation(img_out, flip_type=0, ifonline=1)
    end_time = datetime.datetime.now()
    print('flip0:', end_time - start_time)

    start_time = datetime.datetime.now()
    out_image = flip_operation(img_out, flip_type=1, ifonline=1)
    end_time = datetime.datetime.now()
    print('flip1:', end_time - start_time)

    start_time = datetime.datetime.now()
    out_image = flip_operation(img_out, flip_type=2, ifonline=1)
    end_time = datetime.datetime.now()
    print('flip2:', end_time - start_time)

    start_time = datetime.datetime.now()
    out_image = noise_operation(img_out, ifonline=1)
    end_time = datetime.datetime.now()
    print('noise:', end_time - start_time)

    start_time = datetime.datetime.now()
    out_image = random_erasing_operation(img_out, ifonline=1)
    end_time = datetime.datetime.now()
    print('random_erasing:', end_time - start_time)

def aug_BBox_for_testvis(path):
    for grade_path_name in os.listdir(path):
        grade_path = os.path.join(path, grade_path_name)
        for patient_path_name in os.listdir(grade_path):
            patient_path = os.path.join(grade_path, patient_path_name)
            print('deal with ', patient_path)

            new_patient_path = patient_path

            t1_img = sitk.ReadImage(os.path.join(patient_path, 't1_bbox.nii.gz'))
            t1_img = sitk.GetArrayFromImage(t1_img)

            t2_img = sitk.ReadImage(os.path.join(patient_path, 't2_bbox.nii.gz'))
            t2_img = sitk.GetArrayFromImage(t2_img)

            adc_img = sitk.ReadImage(os.path.join(patient_path, 'adc_bbox.nii.gz'))
            adc_img = sitk.GetArrayFromImage(adc_img)

            all_img = np.concatenate((t1_img, t2_img, adc_img), axis=0)


            if grade_path_name == 'Grade_1':  # 2倍增强, random_erasing
                # for i in range(1):
                #     random_erasing_operation(all_img, new_patient_path, i)
                pass
            elif grade_path_name == 'Grade_2_invasion':  # 11倍增强
                # noise
                noise_operation(all_img, new_patient_path)
                # flip, 0为前后翻转, 1为上下翻转, 2为左右翻转
                for i in range(3):
                    flip_operation(all_img, new_patient_path, i)
                # random_erasing
                for i in range(3):
                    random_erasing_operation(all_img, new_patient_path, i)
                # random_crop
                for i in range(3):
                    random_crop_operation(all_img, new_patient_path, i)
            elif grade_path_name == 'Grade_2_noninvasion':  # 8倍增强
                # noise
                noise_operation(all_img, new_patient_path)
                # flip
                for i in range(2):
                    flip_operation(all_img, new_patient_path, i)
                # random_erasing
                for i in range(2):
                    random_erasing_operation(all_img, new_patient_path, i)
                # random_crop
                for i in range(2):
                    random_crop_operation(all_img, new_patient_path, i)
            else:
                pass

def delete_aug():
    # path = '/mnt/hard_disk/liuwennan/dataset/meningioma_data/Data_processed/bbox/THH_mask/t1+t2+ADC/invasion_or-not/train_aug+crop/Grade_2_invasion'
    # path = '/mnt/hard_disk/liutianling/Invasion/BBox/split/tumor+edema_mask_3folder'
    # path = '/mnt/hard_disk/liutianling/Invasion/BBox/split/tumor+edema_3folders_multicls'
    path = '/mnt/hard_disk/liutianling/Invasion/BBox/split/T1+T2+ADC_random3folder/1-3'
    # for fold_path_name in os.listdir(path):
    #     fold_path = os.path.join(path, fold_path_name)
    #     for traintest_path_name in os.listdir(fold_path):
    #         traintest_path = os.path.join(fold_path, traintest_path_name)
    #         # 仅对训练集数据进行数据增强
    #         if traintest_path_name == 'train':
    #             # for label_path_name in os.listdir(traintest_path):
    #             #     label_path = os.path.join(traintest_path, label_path_name)
    #                 for grade_path_name in os.listdir(traintest_path):
    #                     # if grade_path_name == 'Grade_2_invasion' or grade_path_name == 'Grade_2_noninvasion':
    #                         grade_path = os.path.join(traintest_path, grade_path_name)
    #                         for patient_path_name in os.listdir(grade_path):
    #                             patient_path = os.path.join(grade_path, patient_path_name)
    #                             for modality_path_name in os.listdir(patient_path):
    #                                 modality_name = modality_path_name.split('.nii.gz')[0]
    #                                 if modality_name == 't1_bbox' or modality_name == 't2_bbox' or modality_name == 'adc_bbox':
    #                                     pass
    #                                 else:
    #                                     modality_path = os.path.join(patient_path,modality_path_name)
    #                                     print('delete ',str(modality_path))
    #                                     os.remove(modality_path)
    traintest_path = '/media/ExtHDD02/ltl/Data/MEN/128/cv3folders_multicls/1fold/train_minbalance'
    for grade_path_name in os.listdir(traintest_path):
        # if grade_path_name == 'Grade_2_invasion' or grade_path_name == 'Grade_2_noninvasion':
            grade_path = os.path.join(traintest_path, grade_path_name)
            for patient_path_name in os.listdir(grade_path):
                patient_path = os.path.join(grade_path, patient_path_name)
                for modality_path_name in os.listdir(patient_path):
                    modality_name = modality_path_name.split('.nii.gz')[0]
                    if modality_name == 't1_bbox' or modality_name == 't2_bbox' or modality_name == 'adc_bbox':
                        pass
                    else:
                        modality_path = os.path.join(patient_path,modality_path_name)
                        print('delete ',str(modality_path))
                        os.remove(modality_path)


if __name__ == '__main__':
    # image_size = 224
    # aug_BBox(path='/media/ExtHDD02/ltl/Data/MEN/224/cv3folders_multicls', image_size=image_size)

    # calc_online_aug_time()
    # random_crop:    0:00: 00.029605
    # flip0:          0:00: 00.002873
    # flip1:          0:00: 00.000126
    # flip2:          0:00: 00.000017
    # noise:          0:00: 00.089182
    # random_erasing: 0:00: 00.001583

    # aug_BBox_for_testvis('/media/ExtHDD02/ltl/Data/MEN/cv3folders_multicls/1fold/test4vis')

    delete_aug()






