import os
import cv2
import clip
import random
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# NEV:256, MEL:90
label_list = ['NEV', 'MEL']
num_classes = len(label_list)

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    # RandomContrast,
    RandomGamma,
    # RandomBrightness,
    ShiftScaleRotate,
    RandomBrightnessContrast,
)

shape = [224, 224]

aug = Compose([VerticalFlip(p=0.5),
               HorizontalFlip(p=0.5),
               ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.5,rotate_limit=45,p=0.5),
               RandomRotate90(p=0.5),
               RandomBrightnessContrast(p=0.5),
               #RandomContrast(p=0.5),
               #RandomBrightness(p=0.5),
               #RandomGamma(p=0.5)
               ], p=0.5)

class TB_Dataset(Dataset):

    def __init__(self, subset, data_path, meta_path, args):
        super(TB_Dataset, self).__init__()
        self.args = args
        self.subset = subset
        self.ifonline_data_aug = args.ifonline_data_aug
        self.ifoffline_data_aug = args.ifoffline_data_aug
        self.ifbalanceloader3 = args.ifbalanceloader3
        self.ifbalanceloader4 = args.ifbalanceloader4
        self.meta_data = np.array(pd.read_csv(meta_path, header=None))

        # dataset path
        if 'train' in subset:
            source_path = os.path.join(data_path, 'train')
            print(source_path)
        else:
            source_path = os.path.join(data_path, subset)

        cases = []
        labels = []
        clinic = []
        derm = []
        meta = []
        num_list = [0]*num_classes

        for label_path_name in os.listdir(source_path):
            if label_path_name == 'MEL' or label_path_name == 'NEV':
                label_path = os.path.join(source_path, label_path_name)
                for case_path_name in os.listdir(label_path):
                    case_path = os.path.join(label_path, case_path_name)
                    if self.ifoffline_data_aug:
                        for image_path_name in os.listdir(case_path):
                            if image_path_name.startswith('clinic'):
                                clinic_path = os.path.join(case_path, image_path_name)
                                clinic.append(clinic_path)
                                cases.append(case_path_name)
                                # derm
                                derm_path = os.path.join(case_path, 'derm' + str(image_path_name.split('clinic')[1]))
                                if os.path.exists(derm_path):
                                    derm.append(derm_path)
                                else:
                                    print('not exist ' + derm_path)
                                # meta
                                meta_data = None
                                for i in range(1, self.meta_data.shape[0]):
                                    if case_path_name == str(self.meta_data[i, 0]):
                                        diag_diff_onehot = self.onehot(self.meta_data[i, 10], ['low', 'medium', 'high'])
                                        elevation_onehot = self.onehot(self.meta_data[i, 11], ['palpable', 'nodular', 'flat'])
                                        location_onehot = self.onehot(self.meta_data[i, 12], ['upper limbs', 'head neck', 'back', 'lower limbs', 'chest', 'abdomen', 'buttocks', 'acral', 'genital areas'])
                                        sex_onehot = self.onehot(self.meta_data[i, 13], ['female', 'male'])
                                        management_onehot = self.onehot(self.meta_data[i, 14], ['excision', 'clinical follow up', 'no further examination'])
                                        meta_data = np.array(diag_diff_onehot + elevation_onehot + location_onehot + sex_onehot + management_onehot).astype(np.float32)
                                        meta.append(meta_data)
                                if meta_data is None:
                                    print('meta of case {} is None'.format(case))
                                labels, num_list = self.calc_label(label_path_name, labels, num_list)
                    else:
                        clinic_path = os.path.join(case_path, 'clinic.jpg')
                        if os.path.exists(clinic_path):
                            clinic.append(clinic_path)
                            cases.append(case_path_name)
                        else:
                            print('not exist ' + clinic_path)

                        derm_path = os.path.join(case_path, 'derm.jpg')
                        if os.path.exists(derm_path):
                            derm.append(derm_path)
                        else:
                            print('not exist ' + derm_path)

                        # meta
                        meta_data = None
                        for i in range(1, self.meta_data.shape[0]):
                            if case_path_name == str(self.meta_data[i, 0]):
                                if self.args.clinical_encoder == 'clip':
                                    clinical_text = ['The level of diagnostic difficulty is ',
                                                     'The lesion elevation is ',
                                                     'The lesion location is ',
                                                     'The sex of patient is ',
                                                     'The management for the patient is ']
                                    clinical_text = [clinical_text[j] + str(self.meta_data[i, 10+j]) for j in range(len(clinical_text))]
                                    # print(clinical_text)
                                    clinical_data = clip.tokenize(clinical_text)
                                    meta.append(clinical_data)
                                elif self.args.clinical_encoder == 'mlp':
                                    diag_diff_onehot = self.onehot(self.meta_data[i, 10], ['low', 'medium', 'high'])
                                    elevation_onehot = self.onehot(self.meta_data[i, 11], ['palpable', 'nodular', 'flat'])
                                    location_onehot = self.onehot(self.meta_data[i, 12], ['upper limbs', 'head neck', 'back', 'lower limbs', 'chest', 'abdomen', 'buttocks', 'acral', 'genital areas'])
                                    sex_onehot = self.onehot(self.meta_data[i, 13], ['female', 'male'])
                                    management_onehot = self.onehot(self.meta_data[i, 14], ['excision', 'clinical follow up', 'no further examination'])
                                    clinical_data = np.array(diag_diff_onehot + elevation_onehot + location_onehot + sex_onehot + management_onehot).astype(np.float32)
                                    meta.append(clinical_data)
                        # if meta_data is None:
                        #     print('meta of case {} is None'.format(case))

                        labels, num_list = self.calc_label(label_path_name, labels, num_list)

        self.clinic = clinic
        self.derm = derm
        self.labels = labels
        self.cases = cases
        self.meta = meta

        if subset == 'train':
            print('Num of all samples:', len(labels))
            for i in range(len(label_list)):
                print('Num of label {}: {}'.format(label_list[i], str(num_list[i])))

        if (self.ifbalanceloader3 or self.ifbalanceloader4) and subset == 'train':
            self.list_indices_0 = [index for index, element in enumerate(self.labels) if element == 0]
            self.list_indices_1 = [index for index, element in enumerate(self.labels) if element == 1]
            # self.list_indices_2 = [index for index, element in enumerate(self.labels) if element == 2]
            # self.list_indices_3 = [index for index, element in enumerate(self.labels) if element == 3]
            # self.list_indices_4 = [index for index, element in enumerate(self.labels) if element == 4]

            if self.ifbalanceloader3:
                self.num_batch_each_class = self.args.batch_size_derm7pt // num_classes
                self.use_len = min([x // self.num_batch_each_class * self.num_batch_each_class for x in [len(self.list_indices_0), len(self.list_indices_1)]])
                self.batches = self.use_len // self.num_batch_each_class
                self.list_indices_0 = self.list_indices_0[:self.use_len]
                self.list_indices_1 = self.list_indices_1[:self.use_len]
                # self.list_indices_2 = self.list_indices_2[:self.use_len]
                # self.list_indices_3 = self.list_indices_3[:self.use_len]
                # self.list_indices_4 = self.list_indices_4[:self.use_len]

    def onehot(self, value, value_list):
        onehot_list = [0] * len(value_list)
        for i in range(len(value_list)):
            if value == value_list[i]:
                onehot_list[i] = 1
        return onehot_list

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (shape[0], shape[1]))
        return img

    def load_data(self, clinic_path, derm_path, meta, label, case):
        clinic_img = self.load_image(clinic_path)
        derm_img = self.load_image(derm_path)
        if self.ifonline_data_aug and self.subset == 'train':
            augmented = aug(image=clinic_img, mask=derm_img)
            clinic_img = augmented['image']
            derm_img = augmented['mask']

        clinic_img = np.transpose(clinic_img, (2, 0, 1)).astype('float32') / 255.
        derm_img = np.transpose(derm_img, (2, 0, 1)).astype('float32') / 255.

        img_out = np.concatenate((np.expand_dims(clinic_img, axis=0), np.expand_dims(derm_img, axis=0)), axis=0)
        label = np.array(label)
        return img_out, meta, label, case

    def calc_label(self, label_path_name, labels, num_list=None):
        if label_path_name == 'NEV':
            num_list[0] += 1
            labels.append(0)
        elif label_path_name == 'MEL':
            num_list[1] += 1
            labels.append(1)

        return labels, num_list

    def load_data_mixup_operation(self, list_indices, index_1, index_2):
        img_out_1, meta_data_1, label, case = self.load_data_operation(list_indices, index_1)

        img_out_2, meta_data_2, _, _ = self.load_data_operation(list_indices, index_2)

        img_out = self.args.mixup_scale*img_out_1+(1-self.args.mixup_scale)*img_out_2
        meta_data = self.args.mixup_scale*meta_data_1+(1-self.args.mixup_scale)*meta_data_2
        return img_out, meta_data, label, case


    def __getitem__(self, index):
        if self.ifbalanceloader3 and self.subset == 'train':
            labels = []
            cases = []

            img_out_0, meta_data_0, label_0, case_0 = self.load_data(self.clinic[self.list_indices_0[index]],
                                                                     self.derm[self.list_indices_0[index]],
                                                                     self.meta[self.list_indices_0[index]],
                                                                     self.labels[self.list_indices_0[index]],
                                                                     self.cases[self.list_indices_0[index]])

            img_out_1, meta_data_1, label_1, case_1 = self.load_data(self.clinic[self.list_indices_1[index]],
                                                                     self.derm[self.list_indices_1[index]],
                                                                     self.meta[self.list_indices_1[index]],
                                                                     self.labels[self.list_indices_1[index]],
                                                                     self.cases[self.list_indices_1[index]])

            # img_out_2, meta_data_2, label_2, case_2 = self.load_data(self.clinic[self.list_indices_2[index]],
            #                                                          self.derm[self.list_indices_2[index]],
            #                                                          self.meta[self.list_indices_2[index]],
            #                                                          self.labels[self.list_indices_2[index]],
            #                                                          self.cases[self.list_indices_2[index]])
            #
            # img_out_3, meta_data_3, label_3, case_3 = self.load_data(self.clinic[self.list_indices_3[index]],
            #                                                          self.derm[self.list_indices_3[index]],
            #                                                          self.meta[self.list_indices_3[index]],
            #                                                          self.labels[self.list_indices_3[index]],
            #                                                          self.cases[self.list_indices_3[index]])
            #
            # img_out_4, meta_data_4, label_4, case_4 = self.load_data(self.clinic[self.list_indices_4[index]],
            #                                                          self.derm[self.list_indices_4[index]],
            #                                                          self.meta[self.list_indices_4[index]],
            #                                                          self.labels[self.list_indices_4[index]],
            #                                                          self.cases[self.list_indices_4[index]])
            labels.append(label_0)
            labels.append(label_1)
            # labels.append(label_2)
            # labels.append(label_3)
            # labels.append(label_4)

            cases.append(case_0)
            cases.append(case_1)
            # cases.append(case_2)
            # cases.append(case_3)
            # cases.append(case_4)

            img_out = np.concatenate((np.expand_dims(img_out_0, axis=0),
                                      np.expand_dims(img_out_1, axis=0)), axis=0)
            meta_data = np.concatenate((np.expand_dims(meta_data_0, axis=0),
                                        np.expand_dims(meta_data_1, axis=0)), axis=0)
            return img_out, meta_data, np.array(labels), cases
        elif self.ifbalanceloader4 and self.subset == 'train':
            labels = []
            cases = []
            dwp_value = random.random()

            if self.args.ifmixup_class:
                index_0_1 = random.randint(0, len(self.list_indices_0) - 1)
                index_0_2 = random.randint(0, len(self.list_indices_0) - 1)
                while index_0_1 == index_0_2:
                    index_0_2 = random.randint(0, len(self.list_indices_0) - 1)
                img_out_0, meta_data_0, label_0, case_0 = self.load_data_mixup_operation(self.list_indices_0, index_0_1, index_0_2)

                index_1_1 = random.randint(0, len(self.list_indices_1) - 1)
                index_1_2 = random.randint(0, len(self.list_indices_1) - 1)
                while index_1_1 == index_1_2:
                    index_1_2 = random.randint(0, len(self.list_indices_1) - 1)
                img_out_1, meta_data_1, label_1, case_1 = self.load_data_mixup_operation(self.list_indices_1, index_1_1, index_1_2)

            # elif self.args.ifdwp and dwp_value > 0.6:
            #     img_out_0, meta_data_0, label_0, case_0 = self.load_data(self.clinic[self.list_indices_0[random.randint(0, len(self.list_indices_0) - 1)]],
            #                                                              self.derm[self.list_indices_0[random.randint(0, len(self.list_indices_0) - 1)]],
            #                                                              self.meta[self.list_indices_0[random.randint(0, len(self.list_indices_0) - 1)]],
            #                                                              self.labels[self.list_indices_0[random.randint(0, len(self.list_indices_0) - 1)]],
            #                                                              self.cases[self.list_indices_0[random.randint(0, len(self.list_indices_0) - 1)]])
            #
            #     img_out_1, meta_data_1, label_1, case_1 = self.load_data(self.clinic[self.list_indices_1[random.randint(0, len(self.list_indices_1) - 1)]],
            #                                                              self.derm[self.list_indices_1[random.randint(0, len(self.list_indices_1) - 1)]],
            #                                                              self.meta[self.list_indices_1[random.randint(0, len(self.list_indices_1) - 1)]],
            #                                                              self.labels[self.list_indices_1[random.randint(0, len(self.list_indices_1) - 1)]],
            #                                                              self.cases[self.list_indices_1[random.randint(0, len(self.list_indices_1) - 1)]])

            else:
                index_0 = random.randint(0, len(self.list_indices_0) - 1)
                img_out_0, meta_data_0, label_0, case_0 = self.load_data(self.clinic[self.list_indices_0[index_0]],
                                                                         self.derm[self.list_indices_0[index_0]],
                                                                         self.meta[self.list_indices_0[index_0]],
                                                                         self.labels[self.list_indices_0[index_0]],
                                                                         self.cases[self.list_indices_0[index_0]])


                index_1 = random.randint(0, len(self.list_indices_1) - 1)
                img_out_1, meta_data_1, label_1, case_1 = self.load_data(self.clinic[self.list_indices_1[index_1]],
                                                                         self.derm[self.list_indices_1[index_1]],
                                                                         self.meta[self.list_indices_1[index_1]],
                                                                         self.labels[self.list_indices_1[index_1]],
                                                                         self.cases[self.list_indices_1[index_1]])

            labels.append(label_0)
            labels.append(label_1)

            cases.append(case_0)
            cases.append(case_1)

            img_out = np.concatenate((np.expand_dims(img_out_0, axis=0),
                                      np.expand_dims(img_out_1, axis=0)), axis=0)
            meta_data = np.concatenate((np.expand_dims(meta_data_0, axis=0),
                                        np.expand_dims(meta_data_1, axis=0)), axis=0)
            return img_out, meta_data, np.array(labels), cases
        else:
            img_out, meta_data, label, case = self.load_data(self.clinic[index], self.derm[index], self.meta[index], self.labels[index], self.cases[index])
            return img_out, meta_data, label, case

    def __len__(self):
        if self.ifbalanceloader3 and self.subset == 'train':
            return self.use_len
        elif self.ifbalanceloader4 and self.subset == 'train':
            return len(self.list_indices_0)
        else:
            return len(self.labels)

# class TB_Dataset_onlymodal(Dataset):
#
#     def __init__(self, subset, data_path, ifoffline_data_aug, ifbalanceloader, test_type, image_type, num_classes, ifonline_aug, val=False, args=False):
#         super(TB_Dataset_onlymodal, self).__init__()
#         self.subset = subset
#         self.test_type = test_type
#         self.image_type = image_type
#         self.num_classes = num_classes
#         self.ifonline_aug = ifonline_aug
#         self.args = args
#
#         # dataset path
#         if not val:
#             if 'train' in subset:
#                 source_path = os.path.join(data_path, 'train')
#             elif 'test' in subset:
#                 source_path = os.path.join(data_path, 'test')
#             print(source_path)
#         else:
#             source_path = os.path.join(data_path, 'test')
#
#         patients = []
#         labels = []
#
#         modal = []
#         num_0 = 0
#         num_1 = 0
#         num_2 = 0
#
#         for grade_path_name in os.listdir(source_path):
#             grade_path = os.path.join(source_path, grade_path_name)
#             for patient_path_name in os.listdir(grade_path):
#                 patient_path = os.path.join(grade_path, patient_path_name)
#                 if ifoffline_data_aug:
#                     for nii_path_name in os.listdir(patient_path):
#                         if image_type == 'bbox':
#                             if nii_path_name.startswith(self.args.modal+'_bbox'):
#                                 modal_path = os.path.join(patient_path, nii_path_name)
#                                 modal.append(modal_path)
#                                 patients.append(patient_path_name)
#
#                                 labels, num_0, num_1, num_2 = calc_label(grade_path_name, labels, num_0, num_1, num_2)
#                         else:
#                             print('wrong type!')
#                 else:
#                     if image_type == 'bbox':
#                         modal_path = os.path.join(patient_path, self.args.modal+'_bbox.nii.gz')
#                         if os.path.exists(modal_path):
#                             modal.append(modal_path)
#                             patients.append(patient_path_name)
#
#                             labels, num_0, num_1, num_2 = calc_label(grade_path_name, labels, num_0, num_1, num_2)
#                         else:
#                             print('not exist ' + modal_path)
#
#         if not ifbalanceloader:
#             if not val:
#                 print('Num of all samples:', len(labels))
#                 print('Num of label 0 (Grade_1):', num_0)
#                 print('Num of label 1 (Grade_2_invasion):', num_1)
#                 print('Num of label 2 (Grade_2_noninvasion):', num_2)
#
#
#         self.modal = modal
#         self.labels = labels
#         self.patients = patients
#
#     def __getitem__(self, index):
#
#         img_out, label, patient = load_data_onlymodal(self.modal[index], self.labels[index], self.patients[index], self.subset, self.test_type, self.image_type, self.ifonline_aug)
#         return img_out, label, patient
#
#     def __len__(self):
#         return len(self.labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ifonline_data_aug', type=int, default=0)
    parser.add_argument('--ifoffline_data_aug', type=int, default=0)
    parser.add_argument('--ifbalanceloader3', type=int, default=0)
    parser.add_argument('--ifbalanceloader4', type=int, default=1)
    parser.add_argument('--batch_size_derm7pt', type=int, default=10)
    parser.add_argument('--ifmixup_class', type=int, default=0)
    parser.add_argument('--mixup_scale', type=float, default=0.9)
    args = parser.parse_args()

    data_path = '/media/ExtHDD02/ltl/Data/derm7pt/dealt/augmented'
    meta_path = '/media/ExtHDD02/ltl/Data/derm7pt/release_v0/meta/meta.csv'
    batch_size = 64

    train_set = TB_Dataset('train', data_path, meta_path, args)
    dataloaders = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12, drop_last=True)
    for image, meta_data, label, case in dataloaders:
        print(meta_data)
        print(label)
        print(image.shape)