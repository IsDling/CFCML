import os
import torch
import clip
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_augmentation import online_aug, multimodel_Standard, noise_operation


clip_value = 1200


class TB_Dataset(Dataset):

    def __init__(self, subset, data_path, clinical_path, num_classes, val=0, args=None):
        super(TB_Dataset, self).__init__()
        self.args = args
        self.subset = subset
        self.num_classes = num_classes
        self.clinical_path = clinical_path
        self.ifbalanceloader = args.ifbalanceloader
        self.ifbalanceloader2 = args.ifbalanceloader2
        self.ifbalanceloader3 = args.ifbalanceloader3
        self.ifbalanceloader4 = args.ifbalanceloader4
        self.ifbalanceloader5 = args.ifbalanceloader5
        self.ifoffline_data_aug = args.ifoffline_data_aug
        self.ifonline_data_aug = args.ifonline_data_aug

        if self.args.clinical_encoder == 'clip':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, _ = clip.load("ViT-B/32", device=device)

        # if args.ifclinical:
        clinical = np.array(pd.read_csv(clinical_path, header=None))
        if not self.args.clinical_encoder == 'clip':
            # 对clinical某些列进行归一化
            arr = clinical[1:, 6:]
            arr = arr.astype(float)
            clinical[1:, 6:] = arr / arr.max(axis=0)
        self.clinical = clinical

        # dataset path
        if not val:
            if 'train' in subset:
                if args.ifbalanceloader5:
                    source_path = os.path.join(data_path, 'train_minbalance')
                else:
                    source_path = os.path.join(data_path, 'train')
            else:
                source_path = os.path.join(data_path, 'test')
            print(source_path)
        else:
            if 'train' in subset:
                source_path = os.path.join(data_path, 'train')
            else:
                source_path = os.path.join(data_path, 'test')

        patients = []
        labels = []
        t1 = []
        t2 = []
        adc = []

        num_0 = 0
        num_1 = 0
        num_2 = 0

        if self.ifbalanceloader:
            modal_g1s = []
            modal_g2invs = []
            modal_g2noninvs = []

            patient_g1s = []
            patient_g2invs = []
            patient_g2noninvs = []

        for grade_path_name in os.listdir(source_path):
            grade_path = os.path.join(source_path, grade_path_name)
            for patient_path_name in os.listdir(grade_path):
                patient_path = os.path.join(grade_path, patient_path_name)
                if self.ifoffline_data_aug:
                    for nii_path_name in os.listdir(patient_path):
                        if nii_path_name.startswith('t1_bbox'):
                            t1_path = os.path.join(patient_path, nii_path_name)
                            t1.append(t1_path)
                            patients.append(patient_path_name)
                            # t2
                            t2_path = os.path.join(patient_path, 't2_bbox' + str(nii_path_name.split('t1_bbox')[1]))
                            if os.path.exists(t2_path):
                                t2.append(t2_path)
                            else:
                                print('not exist ' + t2_path)
                            # adc
                            adc_path = os.path.join(patient_path, 'adc_bbox' + str(nii_path_name.split('t1_bbox')[1]))
                            if os.path.exists(adc_path):
                                adc.append(adc_path)
                            else:
                                print('not exist ' + adc_path)

                            labels, num_0, num_1, num_2 = self.calc_label(grade_path_name, labels, num_0, num_1, num_2)
                else:
                    t1_path = os.path.join(patient_path, 't1_bbox.nii.gz')
                    # balanceloader
                    if self.ifbalanceloader and subset == 'train':
                        modal_g1 = []
                        modal_g2inv = []
                        modal_g2noninv = []
                        append = 1
                        if os.path.exists(t1_path):
                            # t1
                            if grade_path_name == 'Grade_1':
                                modal_g1.append(t1_path)
                            elif grade_path_name == 'Grade_2_invasion':
                                modal_g2inv.append(t1_path)
                            elif grade_path_name == 'Grade_2_noninvasion':
                                modal_g2noninv.append(t1_path)
                            # t2
                            t2_path = os.path.join(patient_path, 't2_bbox' + str(t1_path.split('t1_bbox')[1]))
                            if os.path.exists(t2_path):
                                if grade_path_name == 'Grade_1':
                                    modal_g1.append(t2_path)
                                elif grade_path_name == 'Grade_2_invasion':
                                    modal_g2inv.append(t2_path)
                                elif grade_path_name == 'Grade_2_noninvasion':
                                    modal_g2noninv.append(t2_path)
                            else:
                                append = 0
                                print('not exist ' + t2_path)
                            # adc
                            adc_path = os.path.join(patient_path, 'adc_bbox' + str(t1_path.split('t1_bbox')[1]))
                            if os.path.exists(adc_path):
                                if grade_path_name == 'Grade_1':
                                    modal_g1.append(adc_path)
                                elif grade_path_name == 'Grade_2_invasion':
                                    modal_g2inv.append(adc_path)
                                elif grade_path_name == 'Grade_2_noninvasion':
                                    modal_g2noninv.append(adc_path)
                            else:
                                append = 0
                                print('not exist ' + adc_path)
                        else:
                            append = 0
                            print('not exist ' + t1_path)

                        if append == 1:
                            if grade_path_name == 'Grade_1':
                                modal_g1s.append(modal_g1)
                                patient_g1s.append(patient_path_name)
                            elif grade_path_name == 'Grade_2_invasion':
                                modal_g2invs.append(modal_g2inv)
                                patient_g2invs.append(patient_path_name)
                            elif grade_path_name == 'Grade_2_noninvasion':
                                modal_g2noninvs.append(modal_g2noninv)
                                patient_g2noninvs.append(patient_path_name)
                    else:
                        if os.path.exists(t1_path):
                            t1.append(t1_path)
                            patients.append(patient_path_name)

                            # t2
                            t2_path = os.path.join(patient_path, 't2_bbox' + str(t1_path.split('t1_bbox')[1]))
                            if os.path.exists(t2_path):
                                t2.append(t2_path)
                            else:
                                print('not exist ' + t2_path)

                            # adc
                            adc_path = os.path.join(patient_path, 'adc_bbox' + str(t1_path.split('t1_bbox')[1]))
                            if os.path.exists(adc_path):
                                adc.append(adc_path)
                            else:
                                print('not exist ' + adc_path)

                            labels, num_0, num_1, num_2 = self.calc_label(grade_path_name, labels, num_0, num_1, num_2)
                        else:
                            print('not exist ' + t1_path)

        if self.ifbalanceloader and subset == 'train':
                self.modal_g1s = modal_g1s
                self.modal_g2invs = modal_g2invs
                self.modal_g2noninvs = modal_g2noninvs

                self.patient_g1s = patient_g1s
                self.patient_g2invs = patient_g2invs
                self.patient_g2noninvs = patient_g2noninvs

                if not val:
                    print('Num of all samples:', (len(patient_g1s)+len(patient_g2invs)+len(patient_g2noninvs)))
                    print('Num of label 0 (Grade_1):', len(patient_g1s))
                    print('Num of label 1 (Grade_2_invasion):', len(patient_g2invs))
                    print('Num of label 2 (Grade_2_noninvasion):', len(patient_g2noninvs))
        else:
            self.t1 = t1
            self.t2 = t2
            self.adc = adc
            self.labels = labels
            self.patients = patients

            if not val:
                print('Num of all samples:', len(labels))
                print('Num of label 0 (Grade_1):', num_0)
                print('Num of label 1 (Grade_2_invasion):', num_1)
                print('Num of label 2 (Grade_2_noninvasion):', num_2)

            if (self.ifbalanceloader2 or self.ifbalanceloader3 or self.ifbalanceloader4 or self.ifbalanceloader5) and subset == 'train':
                self.list_indices_0 = [index for index, element in enumerate(self.labels) if element == 0]
                self.list_indices_1 = [index for index, element in enumerate(self.labels) if element == 1]
                self.list_indices_2 = [index for index, element in enumerate(self.labels) if element == 2]
                if self.ifbalanceloader2 or self.ifbalanceloader3:
                    self.num_batch_each_class = self.args.batch_size_men//self.num_classes
                    self.use_len = min([x // self.num_batch_each_class * self.num_batch_each_class for x in [len(self.list_indices_0), len(self.list_indices_1), len(self.list_indices_2)]])
                    self.batches = self.use_len//self.num_batch_each_class
                    self.list_indices_0 = self.list_indices_0[:self.use_len]
                    self.list_indices_1 = self.list_indices_1[:self.use_len]
                    self.list_indices_2 = self.list_indices_2[:self.use_len]

    def get_image_offline_aug(self, path):
        img = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(img)
        if self.args.image_encoder == 'res_mamba':
            img = img.transpose((1, 2, 0)).astype(float)
        elif self.args.image_encoder == 'swin_t':
            img = img.astype(float)
        img = np.clip(img, 0, clip_value)
        img = np.expand_dims(img, axis=0)
        return img

    def get_image_online_aug(self, path):
        img = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(img)
        return img

    def onehot(self, value, value_list):
        onehot_list = [0] * len(value_list)
        for i in range(len(value_list)):
            if value == value_list[i]:
                onehot_list[i] = 1
        return onehot_list

    def deal_operation(self, img):
        if self.args.image_encoder == 'res_mamba':
            img = img.transpose((1, 2, 0)).astype(float)
        elif self.args.image_encoder == 'swin_t':
            img = img.astype(float)
        img = np.clip(img, 0, clip_value)
        img = np.expand_dims(img, axis=0)
        return img

    def deal4online_aug(self, image):
        slice_num = int(image.shape[0]/3)
        t1_img = self.deal_operation(image[:slice_num]) # return [1,128,128,24]
        t2_img = self.deal_operation(image[slice_num:slice_num * 2])
        adc_img = self.deal_operation(image[slice_num * 2:])
        return np.concatenate((t1_img, t2_img, adc_img), axis=0)

    def load_data(self, t1_path, t2_path, adc_path, label, patient):
        if (self.ifonline_data_aug and self.subset == 'train') or self.args.test_type == 'ori_noise':
            get_image = self.get_image_online_aug # return [24,128,128]
        else:
            get_image = self.get_image_offline_aug # return [3,128,128,24]

        t1_img = get_image(t1_path)
        t2_img = get_image(t2_path)
        adc_img = get_image(adc_path)

        img_concat = np.concatenate((t1_img, t2_img, adc_img), axis=0)

        if self.subset == 'train' and self.ifonline_data_aug:
            img_on_aug = online_aug(img_concat, image_size=self.args.img_size_men)
            img_out = multimodel_Standard(self.deal4online_aug(img_on_aug))
        elif self.args.test_type == 'ori_noise':
            img_out = multimodel_Standard(self.deal4online_aug(noise_operation(img_concat, ifonline=1)))
        else:
            img_out = multimodel_Standard(img_concat)

        # if self.args.image_encoder == 'swin_t':
        #     img_out = img_out.transpose(0, 3, 1, 2)

        for i in range(1, self.clinical.shape[0]):
            if patient == self.clinical[i, 0]:
                used_clinical = list(self.clinical[i, 4:])
                if self.args.clinical_encoder == 'clip':
                    clinical_text = ['The tumor location in the brain is ',
                                     'The sex of patient is ',
                                     'The age of patient is ',
                                     'The value of apparent diffusion coefficient is ',
                                     'The tumor area in the brain is ',
                                     'The edema area in the brain is']
                    if used_clinical[0] == '1':
                        location_inf = 'convex surface'
                    elif used_clinical[0] == '2':
                        location_inf = 'skull base'
                    elif used_clinical[0] == '3':
                        location_inf = 'other location'
                    else:
                        print('error location_inf')

                    if used_clinical[1] == '0':
                        sex_inf = 'female'
                    elif used_clinical[1] == '1':
                        sex_inf = 'male'
                    else:
                        print('error sex_inf')

                    inf = [location_inf, sex_inf, str(used_clinical[2]), str(used_clinical[3]), str(used_clinical[4]), str(used_clinical[5])]
                    clinical_text = [clinical_text[i]+inf[i] for i in range(len(clinical_text))]

                    clinical_data = clip.tokenize(clinical_text)
                else:
                    location_onehot = self.onehot(used_clinical[0], ['1', '2', '3'])
                    sex_onehot = self.onehot(used_clinical[1], ['0', '1'])
                    clinical_data = np.array(location_onehot + sex_onehot + used_clinical[2:]).astype(np.float32)
        label = np.array(label)
        return img_out, clinical_data, label, patient

    def calc_label(self, grade_path_name, labels, num_0, num_1, num_2):
        if grade_path_name == 'Grade_1':
            labels.append(0)
            num_0 += 1

        if grade_path_name == 'Grade_2_invasion':
            labels.append(1)
            num_1 += 1

        if grade_path_name == 'Grade_2_noninvasion':
            labels.append(2)
            num_2 += 1

        return labels, num_0, num_1, num_2

    def load_data_operation(self, list_indices, index):
        img_out, clinical_data, label, patient = self.load_data(self.t1[list_indices[index]],
                                                                self.t2[list_indices[index]],
                                                                self.adc[list_indices[index]],
                                                                self.labels[list_indices[index]],
                                                                self.patients[list_indices[index]])
        return img_out, clinical_data, label, patient

    def load_data_mixup_operation(self, list_indices, index_1, index_2):
        img_out_1, clinical_data_1, label, patient = self.load_data_operation(list_indices, index_1)
        img_out_2, clinical_data_2, _, _ = self.load_data_operation(list_indices, index_2)

        img_out = self.args.mixup_scale * img_out_1 + (1 - self.args.mixup_scale) * img_out_2
        clinical_data = self.args.mixup_scale * clinical_data_1 + (1 - self.args.mixup_scale) * clinical_data_2
        return img_out, clinical_data, label, patient

    def __getitem__(self, index):
        if self.ifbalanceloader and self.subset == 'train':
            g_s = random.randint(0, 2)
            g_1 = random.randint(0, len(self.modal_g1s) - 1)
            g_2inv = random.randint(0, len(self.modal_g2invs) - 1)
            g_2noninv = random.randint(0, len(self.modal_g2noninvs) - 1)
            
            if g_s == 0:
                modal = self.modal_g1s[g_1]
                patient = self.patient_g1s[g_1]
            elif g_s == 1:
                modal = self.modal_g2invs[g_2inv]
                patient = self.patient_g2invs[g_2inv]
            elif g_s == 2:
                modal = self.modal_g2noninvs[g_2noninv]
                patient = self.patient_g2noninvs[g_2noninv]
            else:
                print('wrong!')
            
            img_out, clinical_data, label, patient = self.load_data(modal[0], modal[1], modal[2], g_s, patient)
            return img_out, clinical_data, label, patient
        elif self.ifbalanceloader2 and self.subset == 'train':
            labels = []
            patients = []
            for i in range(self.num_batch_each_class):
                batch_index = index*self.num_batch_each_class+i
                img_out_0, clinical_data_0, label_0, patient_0 = self.load_data(self.t1[self.list_indices_0[batch_index]], self.t2[self.list_indices_0[batch_index]], self.adc[self.list_indices_0[batch_index]], self.labels[self.list_indices_0[batch_index]], self.patients[self.list_indices_0[batch_index]])
                img_out_1, clinical_data_1, label_1, patient_1 = self.load_data(self.t1[self.list_indices_1[batch_index]], self.t2[self.list_indices_1[batch_index]], self.adc[self.list_indices_1[batch_index]], self.labels[self.list_indices_1[batch_index]], self.patients[self.list_indices_1[batch_index]])
                img_out_2, clinical_data_2, label_2, patient_2 = self.load_data(self.t1[self.list_indices_2[batch_index]], self.t2[self.list_indices_2[batch_index]], self.adc[self.list_indices_2[batch_index]], self.labels[self.list_indices_2[batch_index]], self.patients[self.list_indices_2[batch_index]])
                labels.append(label_0)
                labels.append(label_1)
                labels.append(label_2)
                patients.append(patient_0)
                patients.append(patient_1)
                patients.append(patient_2)

                if i == 0:
                    img_out = np.concatenate((np.expand_dims(img_out_0,axis=0),np.expand_dims(img_out_1,axis=0),np.expand_dims(img_out_2,axis=0)),axis=0)
                    clinical_data = np.concatenate((np.expand_dims(clinical_data_0,axis=0),np.expand_dims(clinical_data_1,axis=0),np.expand_dims(clinical_data_2,axis=0)),axis=0)
                else:
                    img_out = np.concatenate((img_out,np.concatenate((np.expand_dims(img_out_0,axis=0),np.expand_dims(img_out_1,axis=0),np.expand_dims(img_out_2,axis=0)),axis=0)),axis=0)
                    clinical_data = np.concatenate((clinical_data,np.concatenate((np.expand_dims(clinical_data_0,axis=0),np.expand_dims(clinical_data_1,axis=0),np.expand_dims(clinical_data_2,axis=0)),axis=0)),axis=0)

            return img_out, clinical_data, np.array(labels), patients
        elif self.ifbalanceloader3 and self.subset == 'train':
            labels = []
            patients = []

            img_out_0, clinical_data_0, label_0, patient_0 = self.load_data(self.t1[self.list_indices_0[index]], self.t2[self.list_indices_0[index]], self.adc[self.list_indices_0[index]], self.labels[self.list_indices_0[index]], self.patients[self.list_indices_0[index]])
            img_out_1, clinical_data_1, label_1, patient_1 = self.load_data(self.t1[self.list_indices_1[index]], self.t2[self.list_indices_1[index]], self.adc[self.list_indices_1[index]], self.labels[self.list_indices_1[index]], self.patients[self.list_indices_1[index]])
            img_out_2, clinical_data_2, label_2, patient_2 = self.load_data(self.t1[self.list_indices_2[index]], self.t2[self.list_indices_2[index]], self.adc[self.list_indices_2[index]], self.labels[self.list_indices_2[index]], self.patients[self.list_indices_2[index]])
            labels.append(label_0)
            labels.append(label_1)
            labels.append(label_2)
            patients.append(patient_0)
            patients.append(patient_1)
            patients.append(patient_2)

            img_out = np.concatenate((np.expand_dims(img_out_0,axis=0),np.expand_dims(img_out_1,axis=0),np.expand_dims(img_out_2,axis=0)),axis=0)
            clinical_data = np.concatenate((np.expand_dims(clinical_data_0,axis=0),np.expand_dims(clinical_data_1,axis=0),np.expand_dims(clinical_data_2,axis=0)),axis=0)
            return img_out, clinical_data, np.array(labels), patients
        elif self.ifbalanceloader4 and self.subset == 'train':
            labels = []
            patients = []

            # label 0
            img_out_0, clinical_data_0, label_0, patient_0 = self.load_data_operation(self.list_indices_0, index)

            if self.args.ifmixup_class:
                # label 1
                index_1_1 = random.randint(0, len(self.list_indices_1) - 1)
                index_1_2 = random.randint(0, len(self.list_indices_1) - 1)
                while index_1_1 == index_1_2:
                    index_1_2 = random.randint(0, len(self.list_indices_1) - 1)
                img_out_1, clinical_data_1, label_1, patient_1 = self.load_data_mixup_operation(self.list_indices_1, index_1_1, index_1_2)

                # label 2
                index_2_1 = random.randint(0, len(self.list_indices_2) - 1)
                index_2_2 = random.randint(0, len(self.list_indices_2) - 1)
                while index_2_1 == index_2_2:
                    index_2_2 = random.randint(0, len(self.list_indices_2) - 1)
                img_out_2, clinical_data_2, label_2, patient_2 = self.load_data_mixup_operation(self.list_indices_2, index_2_1, index_2_2)

            else:
                # label 1
                index_1 = random.randint(0, len(self.list_indices_1) - 1)
                img_out_1, clinical_data_1, label_1, patient_1 = self.load_data_operation(self.list_indices_1, index_1)

                # label 2
                index_2 = random.randint(0, len(self.list_indices_2) - 1)
                img_out_2, clinical_data_2, label_2, patient_2 = self.load_data_operation(self.list_indices_2, index_2)

            labels.append(label_0)
            labels.append(label_1)
            labels.append(label_2)
            patients.append(patient_0)
            patients.append(patient_1)
            patients.append(patient_2)

            img_out = np.concatenate((np.expand_dims(img_out_0, axis=0), np.expand_dims(img_out_1, axis=0), np.expand_dims(img_out_2, axis=0)), axis=0)

            # if self.args.clinical_encoder == 'clip':
            #     clinical_data = [clinical_data_0, clinical_data_1, clinical_data_2]
            # else:
            clinical_data = np.concatenate((np.expand_dims(clinical_data_0, axis=0),
                                            np.expand_dims(clinical_data_1, axis=0),
                                            np.expand_dims(clinical_data_2, axis=0)), axis=0)
            return img_out, clinical_data, np.array(labels), patients
        elif self.ifbalanceloader5 and self.subset == 'train':
            labels = []
            patients = []

            index_0 = random.randint(0, len(self.list_indices_0) - 1)
            img_out_0, clinical_data_0, label_0, patient_0 = self.load_data(self.t1[self.list_indices_0[index_0]],
                                                                            self.t2[self.list_indices_0[index_0]],
                                                                            self.adc[self.list_indices_0[index_0]],
                                                                            self.labels[self.list_indices_0[index_0]],
                                                                            self.patients[self.list_indices_0[index_0]])

            index_1 = random.randint(0, len(self.list_indices_1) - 1)
            img_out_1, clinical_data_1, label_1, patient_1 = self.load_data(self.t1[self.list_indices_1[index_1]],
                                                                            self.t2[self.list_indices_1[index_1]],
                                                                            self.adc[self.list_indices_1[index_1]],
                                                                            self.labels[self.list_indices_1[index_1]],
                                                                            self.patients[self.list_indices_1[index_1]])

            index_2 = random.randint(0, len(self.list_indices_2) - 1)
            img_out_2, clinical_data_2, label_2, patient_2 = self.load_data(self.t1[self.list_indices_2[index_2]],
                                                                            self.t2[self.list_indices_2[index_2]],
                                                                            self.adc[self.list_indices_2[index_2]],
                                                                            self.labels[self.list_indices_2[index_2]],
                                                                            self.patients[self.list_indices_2[index_2]])
            labels.append(label_0)
            labels.append(label_1)
            labels.append(label_2)
            patients.append(patient_0)
            patients.append(patient_1)
            patients.append(patient_2)

            img_out = np.concatenate((np.expand_dims(img_out_0, axis=0), np.expand_dims(img_out_1, axis=0),
                                      np.expand_dims(img_out_2, axis=0)), axis=0)
            clinical_data = np.concatenate((np.expand_dims(clinical_data_0, axis=0),
                                            np.expand_dims(clinical_data_1, axis=0),
                                            np.expand_dims(clinical_data_2, axis=0)), axis=0)

            return img_out, clinical_data, np.array(labels), patients
        else:
            img_out, clinical_data, label, patient = self.load_data(self.t1[index], self.t2[index], self.adc[index], self.labels[index], self.patients[index])
            return img_out, clinical_data, label, patient

    def __len__(self):
        if self.ifbalanceloader and self.subset == 'train':
            return len(self.modal_g1s) * self.args.num_aug
        elif self.ifbalanceloader2 and self.subset == 'train':
            return self.batches
        elif self.ifbalanceloader3 and self.subset == 'train':
            return self.use_len
        elif self.ifbalanceloader4 and self.subset == 'train':
            return len(self.list_indices_0)
        elif self.ifbalanceloader5 and self.subset == 'train':
            return len(self.list_indices_0)*self.args.num_aug
        else:
            return len(self.labels)
