import torch
import torch.nn as nn
import myKit
import warnings
warnings.filterwarnings("ignore")

""""还未进行重构的训练函数"""

# loss = nn.CrossEntropyLoss(reduction="none")
loss_fn = nn.L1Loss(reduction='sum')

# criterion = nn.CrossEntropyLoss(reduction='none')
if __name__ == '__main__':

    net = myKit.get_net(isEnsemble=False)
    lr = 1e-4
    batch_size = 8
    num_epochs = 5
    weight_decay = 0
    lr_period = 2
    lr_decay = 0.8
    # bone_dir = os.path.join('..', 'data', 'archive', 'testDataset')
    bone_dir = "../archive"
    csv_name = "boneage-training-dataset.csv"
    train_df, valid_df = myKit.split_data(bone_dir, csv_name, 10, 0.1, 10)
    train_set, val_set = myKit.create_data_loader(train_df, valid_df)
    torch.set_default_tensor_type('torch.FloatTensor')
    myKit.map_fn(net=net, train_dataset=train_set, valid_dataset=val_set, num_epochs=num_epochs, lr=lr, wd=weight_decay, lr_period=lr_period, lr_decay=lr_decay,loss_fn=loss_fn, batch_size=batch_size, model_path="model.pth", record_path="RECORD.csv")
