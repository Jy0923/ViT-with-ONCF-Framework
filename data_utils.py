import numpy as np 
import pandas as pd 
import scipy.sparse as sp

from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data

def load_all(data_path: str,
             test_size: float = 0.3):
    """ We load all the three file here to save time in each epoch. """
    """
    data_path: 데이터 경로
    test_size: test data 비율
    """
    train_data = pd.read_csv(
        data_path, dtype={0: np.int32, 1: np.int32})

    user_num = train_data['회원번호'].max() + 1
    item_num = train_data['책제목'].max() + 1

    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
    
    # 데이터 분할
    train_data, test_data = train_test_split(train_data, test_size=test_size, shuffle=True, random_state=0)

    return train_data, test_data, user_num, item_num, train_mat

def load_aux(data_path: str,
            aux_col: str,
             test_size: float = 0.3):
    """ We load all the three file here to save time in each epoch. """
    """
    data_path: 데이터 경로
    aux_col: auxiliary 정보가 있는 column명
    test_size: test data 비율
    """
    train_data = pd.read_csv(
        data_path, dtype={0: np.int32, 1: np.int32})

    user_num = train_data['회원번호'].max() + 1
    item_num = train_data[aux_col].max() + 1

    train_data = train_data.values.tolist()

    # load auxiliary information as a dok matrix
    mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        mat[x[0], x[1]] = 1.0
    
    return mat

class CustomDataset(data.Dataset):
    def __init__(self, data_path_main : str,
                 data_path_aux_user = None, aux_col_user = None,
                 data_path_aux_item = None, aux_col_item = None,
                 test_size : float = 0.3, num_ng=0, is_training=None):
        super(CustomDataset, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        """
        data_path_main: main data(rating matrix) 데이터 경로
        data_path_aux_user: user auxiliary information 데이터 경로(들) -> 리스트 형태
        aux_col_user: user axiliary information 정보가 있는 column명(들) -> 리스트 형태
        data_path_aux_item: item auxiliary information 데이터 경로(들) -> 리스트 형태
        aux_col_item: item axiliary information 정보가 있는 column명(들) -> 리스트 형태
        test_size: test data 비율
        num_ng: negative sampling 비율 (vs positive sample)
        is_training: training 여부
        """

        # loading main data
        train_data, test_data, user_num, item_num, train_mat = load_all(data_path_main, test_size)
        
        # loading user auxiliary information data
        try:
            for i in range(len(data_path_aux_user)):
                self.globals()[f'mat_{aux_col_user[i]}'] = (data_path_aux_user[i], aux_col_user[i], test_size)
        except:
            pass
        
        # loading item auxiliary information data
        try:
            for i in range(len(data_path_aux_item)):
                self.globals()[f'mat_{aux_col_item[i]}'] = (data_path_aux_item[i], aux_col_item[i], test_size)
        except:
            pass
        
        # 학습 여부에 따라 features 변수에 알맞는 데이터 할당
        if is_training == True:
            features = train_data
        elif is_training == False:
            features = test_data
        
        self.features_ps = features
        self.num_item = item_num
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
        """negative sampling"""
        assert self.is_training, 'no need to sampling when testing'

        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        """length of data"""
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training \
                    else self.features_ps
        labels = self.labels_fill if self.is_training \
                    else self.labels

        # user, item, label
        user = torch.LongTensor(features[idx][0])
        item = torch.LongTensor(features[idx][1])
        label_main = torch.LongTensor(labels[idx])
        
        results = {'user_id':user, 'item_id':item, 'target_main':label_main}
        
        # auxiliary information
        for i in range(len(data_path_aux_user)):
            results.update( {f'target_user_aux_{aux_col_user[i]}' : torch.LongTensor(self.globals()[f'mat_{aux_col_user[i]}'][user])} )
        for i in range(len(data_path_aux_item)):
            results.update( {f'target_item_aux_{aux_col_item[i]}' : torch.LongTensor(self.globals()[f'mat_{aux_col_item[i]}'][item])} )
        
        return results