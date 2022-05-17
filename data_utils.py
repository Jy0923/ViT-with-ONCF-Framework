import numpy as np 
import pandas as pd 
import scipy.sparse as sp

from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data

def load_all(data_path: str,
             test_size: float = 0.3):
    """ We load all the three file here to save time in each epoch. """
    train_data = pd.read_csv(
        data_path, dtype={0: np.int32, 1: np.int32})

    user_num = train_data['회원번호'].max() + 1
    item_num = train_data['책제목'].max() + 1

    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    train_data, test_data = train_test_split(train_data, test_size=test_size, shuffle=True, random_state=0)

    return train_data, test_data, user_num, item_num, train_mat

def load_aux(data_path: str,
            aux_col: str,
             test_size: float = 0.3):
    """ We load all the three file here to save time in each epoch. """
    train_data = pd.read_csv(
        data_path, dtype={0: np.int32, 1: np.int32})

    user_num = train_data['회원번호'].max() + 1
    item_num = train_data[aux_col].max() + 1

    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        mat[x[0], x[1]] = 1.0
        
    train_data, test_data = train_test_split(train_data, test_size=test_size, shuffle=True, random_state=0)
    
    return train_data, test_data, mat

class CustomDataset(data.Dataset):
    def __init__(self, features, num_item,
                 train_mat=None, num_ng=0, is_training=None):
        super(CustomDataset, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
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
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training \
                    else self.features_ps
        labels = self.labels_fill if self.is_training \
                    else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item ,label
    
def Make_DataSet(data_path_main : str,
                 test_size : float = 0.3,
                 num_ng: int = 4,
                 batch_size: int = 32,
                 data_path_user_aux = None,
                 cols_user_aux = None,
                 data_path_item_aux = None,
                 cols_item_aux = None):
    # main data 생성
    train_data_main, test_data_main, user_num_main, item_num_main, train_mat_main\
        = load_all(data_path_main, test_size)
    
    train_dataset_main = CustomDataset(train_data_main, item_num_main, train_mat_main, num_ng, True)
    test_dataset_main = CustomDataset(test_data_main, item_num_main, train_mat_main, 0, False)
    train_loader_main = data.DataLoader(train_dataset_main, batch_size=batch_size, shuffle=True)
    test_loader_main = data.DataLoader(test_dataset_main, batch_size=batch_size, shuffle=False) # 여기 batch_size 어떻게 해야 할까요?
    
    # user auxiliary data 생성
    train_dataset_user_auxes = []
    test_dataset_user_auxes = []
    train_loader_user_auxes = []
    test_loader_user_auxes = []
    try:
        for i in range(len(data_path_user_aux)):
            # 데이터, 행렬 불러오기
            globals()[f'train_data_{cols_user_aux[i]}'], globals()[f'test_data_{cols_user_aux[i]}'],\
            globals()[f'mat_{cols_user_aux[i]}']\
                = load_aux(data_path_user_aux[i], cols_user_aux[i], test_size)
            # target 생성
            globals()[f'train_targets_{cols_user_aux[i]}']\
                = sp.vstack([globals()[f'mat_{cols_user_aux[i]}'][user] for user, aux in globals()[f'train_data_{cols_user_aux[i]}']])
            globals()[f'test_targets_{cols_user_aux[i]}']\
                = sp.vstack([globals()[f'mat_{cols_user_aux[i]}'][user] for user, aux in globals()[f'test_data_{cols_user_aux[i]}']])
            mat_tr = globals()[f'train_targets_{cols_user_aux[i]}']
            mat_te = globals()[f'test_targets_{cols_user_aux[i]}']
            mat_tr = torch.sparse.LongTensor(torch.LongTensor([mat_tr.tocoo().row.tolist(), mat_tr.tocoo().col.tolist()]),
                           torch.LongTensor(mat_tr.tocoo().data.astype(np.int32)))
            mat_te = torch.sparse.LongTensor(torch.LongTensor([mat_te.tocoo().row.tolist(), mat_te.tocoo().col.tolist()]),
                           torch.LongTensor(mat_te.tocoo().data.astype(np.int32)))
            # dataset 생성
            globals()[f'train_dataset_{cols_user_aux[i]}']\
            = data.TensorDataset(torch.tensor(globals()[f'train_data_{cols_user_aux[i]}']), mat_tr)
            globals()[f'test_dataset_{cols_user_aux[i]}']\
            = data.TensorDataset(torch.tensor(globals()[f'test_data_{cols_user_aux[i]}']), mat_te)
            # data loader 생성
            globals()[f'train_loader_{cols_user_aux[i]}'] = data.DataLoader(globals()[f'train_dataset_{cols_user_aux[i]}'], batch_size=batch_size, shuffle=True)
            globals()[f'test_loader_{cols_user_aux[i]}'] = data.DataLoader(globals()[f'test_dataset_{cols_user_aux[i]}'], batch_size=batch_size, shuffle=False) # 여기 batch_size 어떻게 해야 할까요?
            train_dataset_user_auxes.append(globals()[f'train_dataset_{cols_user_aux[i]}'])
            test_dataset_user_auxes.append(globals()[f'test_dataset_{cols_user_aux[i]}'])
            train_loader_user_auxes.append(globals()[f'train_loader_{cols_user_aux[i]}'])
            test_loader_user_auxes.append(globals()[f'test_loader_{cols_user_aux[i]}'])
    except:
        pass
    
    # item auxiliary data 생성
    train_dataset_item_auxes = []
    test_dataset_item_auxes = []
    train_loader_item_auxes = []
    test_loader_item_auxes = []
    try:
        for i in range(len(data_path_item_aux)):
            # 데이터, 행렬 불러오기
            globals()[f'train_data_{cols_item_aux[i]}'], globals()[f'test_data_{cols_item_aux[i]}'],\
            globals()[f'mat_{cols_item_aux[i]}']\
                = load_aux(data_path_item_aux[i], cols_item_aux[i], test_size)
            # target 생성
            globals()[f'train_targets_{cols_item_aux[i]}']\
                = sp.vstack([globals()[f'mat_{cols_item_aux[i]}'][user] for user, aux in globals()[f'train_data_{cols_item_aux[i]}']])
            globals()[f'test_targets_{cols_item_aux[i]}']\
                = sp.vstack([globals()[f'mat_{cols_item_aux[i]}'][user] for user, aux in globals()[f'test_data_{cols_item_aux[i]}']])
            mat_tr = globals()[f'train_targets_{cols_item_aux[i]}']
            mat_te = globals()[f'test_targets_{cols_item_aux[i]}']
            mat_tr = torch.sparse.LongTensor(torch.LongTensor([mat_tr.tocoo().row.tolist(), mat_tr.tocoo().col.tolist()]),
                           torch.LongTensor(mat_tr.tocoo().data.astype(np.int32)))
            mat_te = torch.sparse.LongTensor(torch.LongTensor([mat_te.tocoo().row.tolist(), mat_te.tocoo().col.tolist()]),
                           torch.LongTensor(mat_te.tocoo().data.astype(np.int32)))
            # dataset 생성
            globals()[f'train_dataset_{cols_item_aux[i]}']\
            = data.TensorDataset(torch.tensor(globals()[f'train_data_{cols_item_aux[i]}']), mat_tr)
            globals()[f'test_dataset_{cols_item_aux[i]}']\
            = data.TensorDataset(torch.tensor(globals()[f'test_data_{cols_item_aux[i]}']), mat_te)
            # data loader 생성
            globals()[f'train_loader_{cols_item_aux[i]}'] = data.DataLoader(globals()[f'train_dataset_{cols_item_aux[i]}'], batch_size=batch_size, shuffle=True)
            globals()[f'test_loader_{cols_item_aux[i]}'] = data.DataLoader(globals()[f'test_dataset_{cols_item_aux[i]}'], batch_size=batch_size, shuffle=False) # 여기 batch_size 어떻게 해야 할까요?
            train_dataset_item_auxes.append(globals()[f'train_dataset_{cols_item_aux[i]}'])
            test_dataset_item_auxes.append(globals()[f'test_dataset_{cols_item_aux[i]}'])
            train_loader_item_auxes.append(globals()[f'train_loader_{cols_item_aux[i]}'])
            test_loader_item_auxes.append(globals()[f'test_loader_{cols_item_aux[i]}'])
    except:
        pass
    
    results = {'train_dataset_main':train_dataset_main, 'test_dataset_main':test_dataset_main,
              'train_loader_main':train_loader_main, 'test_loader_main':test_loader_main,
              'train_dataset_user_auxes':train_dataset_user_auxes, 'test_dataset_user_auxes':test_dataset_user_auxes,
              'train_loader_user_auxes':train_loader_user_auxes, 'test_loader_user_auxes':test_loader_user_auxes,
              'train_dataset_item_auxes':train_dataset_item_auxes, 'test_dataset_item_auxes':test_dataset_item_auxes,
              'train_loader_item_auxes':train_loader_item_auxes, 'test_loader_item_auxes':test_loader_item_auxes}
    return results