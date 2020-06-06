# -*- coding: utf-8 -*-
'''
@time: 2019/7/23 19:42

@ author: javis
'''
import torch, time, os, shutil
import models, utils
import numpy as np
import pandas as pd
from tensorboard_logger import Logger
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import ECGDataset, my_collate_fn
from config2 import config
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(41)
torch.cuda.manual_seed(41)


# 保存当前模型的权重，并且更新最佳的模型权重
def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, config.current_w)
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)


def train_epoch(model, optimizer, criterion, train_dataloader, epoch, lr, best_f1, show_interval=10):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    tq = tqdm.tqdm(total=len(train_dataloader)*config.batch_size)
    tq.set_description('epoch %d, lr %.4f, best_f:%.4f' % (epoch, lr, best_f1))

    for i, (inputs, target) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        f1 = utils.calc_f1(target, torch.sigmoid(output))
        f1_meter += f1
        tq.update(config.batch_size)
        tq.set_postfix(loss="%.4f   f1:%.3f" % (loss.item(), f1))
    tq.close()
        #if it_count != 0 and it_count % show_interval == 0:
#        print("%d,loss:%.3e f1:%.3f" % (it_count, loss.item(), f1), end='\r')
    return loss_meter / it_count, f1_meter / it_count


def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    f1_meter, loss_meter, it_count = 0, 0, 0
    with torch.no_grad():
        if torch.cuda.is_available():
            label_all = torch.Tensor().cuda()
            pred_all = torch.Tensor().cuda()
        else:
            label_all = torch.Tensor()
            pred_all = torch.Tensor()
        tq = tqdm.tqdm(total=len(val_dataloader) * config.batch_size)
        for inputs, target in val_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            it_count += 1
            label_all = torch.cat((label_all, target), 0)
            pred_all = torch.cat((pred_all, output), 0)
            tq.update(config.batch_size)
        tq.close()
        output = pred_all
        target = label_all
        loss = criterion(output, target)
        loss_meter = loss.item()

        output = torch.sigmoid(output)
        f1 = utils.calc_f1(target, output, threshold)
        # f1_meter += f1
    return loss_meter, f1


def train(args):
    # model
    print(args.model_name)
    config.train_data = config.train_data + str(args.fold) + '.pth'
    config.model_name = args.model_name
    model = getattr(models, config.model_name)()
    # if args.ckpt and not args.resume:
    #     state = torch.load(args.ckpt, map_location='cpu')
    #     model.load_state_dict(state['state_dict'])
    #     print('train with pretrained weight val_f1', state['f1'])
    model = model.to(device)
    # data
    train_dataset = ECGDataset(data_path=config.train_data, train=True)
    train_dataloader = DataLoader(train_dataset, collate_fn=my_collate_fn, batch_size=config.batch_size, shuffle=True, num_workers=6)
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=6)
    print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=config.lr)
    w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
    criterion = utils.WeightedMultilabel(w)
    # 模型保存文件夹
    model_save_dir = '%s/%s' % (config.ckpt, config.model_name + '_' + str(args.fold))
    args.ckpt = model_save_dir
    # if args.ex: model_save_dir += args.ex
    best_f1 = -1
    lr = config.lr
    start_epoch = 1
    stage = 1
    # 从上一个断点，继续训练
    if args.resume:
        if os.path.exists(args.ckpt):  # 这里是存放权重的目录
            # model_save_dir = args.ckpt
            current_w = torch.load(os.path.join(args.ckpt, config.current_w))
            best_w = torch.load(os.path.join(model_save_dir, config.best_w))
            best_f1 = best_w['best_f']
            start_epoch = current_w['epoch'] + 1
            lr = current_w['lr']
            stage = current_w['stage']
            model.load_state_dict(current_w['state_dict'])
            # 如果中断点恰好为转换stage的点
            if start_epoch - 1 in config.stage_epoch:
                stage += 1
                lr /= config.lr_decay
                utils.adjust_learning_rate(optimizer, lr)
                model.load_state_dict(best_w['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
    logger = Logger(logdir=model_save_dir, flush_secs=2)
    # =========>开始训练<=========
    val_loss = 10
    val_f1 = -1
    # print(lr)
    # lr = config.lr
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader, epoch, lr, best_f1, show_interval=100)
        # if epoch % 2 == 1:
        val_loss, val_f1 = val_epoch(model, criterion, val_dataloader)
        print('#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s\n'
              % (epoch, stage, train_loss, train_f1, val_loss, val_f1, utils.print_time_cost(since)))
        logger.log_value('train_loss', train_loss, step=epoch)
        logger.log_value('train_f1', train_f1, step=epoch)
        logger.log_value('val_loss', val_loss, step=epoch)
        logger.log_value('val_f1', val_f1, step=epoch)
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,
                 'stage': stage, "best_f": best_f1}
        if best_f1 < val_f1:
            save_ckpt(state, best_f1 < val_f1, model_save_dir)
            print('save best')
        best_f1 = max(best_f1, val_f1)

        if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay
            best_w = os.path.join(model_save_dir, config.best_w)
            save_ckpt(state, False, model_save_dir)
#            model.load_state_dict(torch.load(best_w)['state_dict'])
            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)

#用于测试加载模型
def val(args):
    
    config.model_name = args.model_name
    print(config.model_name)
    model_save_dir = '%s/%s' % (config.ckpt, config.model_name + '_' + str(args.fold))
    args.ckpt = model_save_dir
    config.train_data = config.train_data + str(args.fold) + '.pth'
    list_threhold = [0.5]
    model = getattr(models, config.model_name)()
    if args.ckpt: model.load_state_dict(torch.load(
            os.path.join(model_save_dir, config.best_w), map_location='cpu')['state_dict'])
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)
    for threshold in list_threhold:
        val_loss, val_f1 = val_epoch(model, criterion, val_dataloader, threshold)
        print('threshold %.2f val_loss:%0.3e val_f1:%.3f\n' % (threshold, val_loss, val_f1))

#提交结果使用
def test(args):
    from dataset import ECGDataset_test
    from data_process import name2index
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}

    sub_file = 'result.txt'
    fout = open(sub_file, 'w', encoding='utf-8')
    config.test_label = args.test_label
    config.test_dir = args.test_dir
    path_all = []
    for line in open(config.test_label, encoding='utf-8'):
    #            fout.write(line.strip('\n'))
        id = line.split('\t')[0]
        file_path = os.path.join(config.test_dir, id)
        path_all.append(file_path)
    test_dataset = ECGDataset_test(path_all)
    test_load = DataLoader(test_dataset, batch_size=64, num_workers=6, 
                           shuffle=False)
    
    out = np.zeros((len(test_dataset), 34))
    print(len(test_dataset))
    for fold in range(5):
        model_save_dir = '%s/%s' % (config.ckpt, config.model_name + '_' + str(fold))
        print(model_save_dir)
        model = getattr(models, 'myecgnet1')()
        model.load_state_dict(torch.load(os.path.join(model_save_dir, config.best_w), 
                                         map_location='cpu')['state_dict'])
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            
            
            if torch.cuda.is_available():
                pred_all = torch.Tensor().cuda()
            else:
                pred_all = torch.Tensor()
                
            for inputs, target in tqdm.tqdm(test_load):
                inputs = inputs.to(device)
                output, _ = model(inputs)
                output = torch.sigmoid(output)
                pred_all = torch.cat((pred_all, output), 0)
            out += pred_all.cpu().numpy()
        
    out = out / 5
    num = 0
    for line in open(config.test_label, encoding='utf-8'):
        fout.write(line.strip('\n'))
        output = out[num]
        ixs = [i for i, out in enumerate(output) if i not in [0, 2, 6, 7, 8, 14, 15, 16, 19, 21, 23, 25, 32, 33] and out > 0.5]
        for i in ixs:
            fout.write("\t" + idx2name[i])
        fout.write('\n')
        num = num + 1
    fout.close()
    

#def get_feature(args):
#    from dataset import ECGDataset_test
#    config.model_name = args.model_name
#    model_save_dir = '%s/%s' % (config.ckpt, config.model_name + '_' + str(args.fold))
##    utils.mkdirs(config.sub_dir)
#    # model
#    model = getattr(models, config.model_name)()
#    model.load_state_dict(torch.load(os.path.join(model_save_dir, config.best_w), 
#                                     map_location='cpu')['state_dict'])
#    model = model.to(device)
#    model.eval()
##    config.test_label = 'hf_round2_train.txt'
##    config.test_dir = config.train_dir
#    path_all = []
#    filename = []
#    with torch.no_grad():
#        for line in open(config.test_label, encoding='utf-8'):
#            #            fout.write(line.strip('\n'))
#            id = line.split('\t')[0]
#            file_path = os.path.join(config.test_dir, id)
#            filename.append(id)
#            path_all.append(file_path)
#        test_dataset = ECGDataset_test(path_all)
#        test_load = DataLoader(test_dataset, batch_size=128, num_workers=6,
#                               shuffle=False)
#
#        if torch.cuda.is_available():
#            pred_all = torch.Tensor().cuda()
#            prob_all = torch.Tensor().cuda()
#        else:
#            pred_all = torch.Tensor()
#            prob_all = torch.Tensor()
#        for inputs, target in tqdm.tqdm(test_load):
#            inputs = inputs.to(device)
#            prob, feature = model(inputs)
#            #            output = torch.sigmoid(out)
#            pred_all = torch.cat((pred_all, feature), 0)
#            prob_all = torch.cat((prob_all, prob), 0)
#        out = pred_all.cpu().numpy()
#        df = pd.DataFrame(out)
#        df['filename'] = filename
#        df.to_csv('lgb/DL_feature/' + str(args.fold) + '_test.csv', index=None)
#
#        out = prob_all.cpu().numpy()
#        df = pd.DataFrame(out)
#        df['filename'] = filename
#        df.to_csv('lgb/DL_feature/prob_' + str(args.fold) + '.csv', index=None)


def get_feature(args):
#    device = 'cpu'
    from dataset import ECGDataset_test
    fold = [0, 1, 2, 3, 4]
    model_all = []
#    for i in fold:
#        config.model_name = 'resnet34'
#        model_save_dir = '%s/%s' % (config.ckpt, config.model_name + '_' + str(i))
#        model = getattr(models, config.model_name)()
#        model.load_state_dict(torch.load(os.path.join(model_save_dir, config.best_w), 
#                                     map_location='cpu')['state_dict'])
#        model = model.to(device)
#        model.eval()
#        model_all.append(model)
        
    for i in fold:
        config.model_name = 'myecgnet'
        model_save_dir = '%s/%s' % (config.ckpt2, config.model_name + '_' + str(i))
#        if i in [0, 1, 4]:
#            print('load')
#            model = getattr(models, config.model_name)()
#        else:
        model = getattr(models, 'myecgnet1')()
        
#        from torchsummary import summary
#        model = model.to(device)
#        summary(model, (12, 5000))
        
        model.load_state_dict(torch.load(os.path.join(model_save_dir, config.best_w), 
                                     map_location='cpu')['state_dict'])
        model = model.to(device)
        model.eval()
        model_all.append(model)
    
    path_all = []
    filename = []
    with torch.no_grad():
        for line in open(config.test_label, encoding='utf-8'):
            #            fout.write(line.strip('\n'))
            id = line.split('\t')[0]
            file_path = os.path.join(config.test_dir, id)
            filename.append(id)
            path_all.append(file_path)
        test_dataset = ECGDataset_test(path_all)
        test_load = DataLoader(test_dataset, batch_size=128, num_workers=6,
                               shuffle=False)
        pred_list = []
        prob_list = []
        
        for i in range(5):
            if torch.cuda.is_available():
                pred_list.append(torch.Tensor().cuda())
                prob_list.append(torch.Tensor().cuda())
            else:
                pred_list.append(torch.Tensor())
                prob_list.append(torch.Tensor())
                
#        for i in range(10):
#            pred_list.append(pred_all.copy())
#            prob_list.append(prob_all.copy())
            
        for inputs, target in tqdm.tqdm(test_load):
            inputs = inputs.to(device)
            
            for i in range(5):
                prob, feature = model_all[i](inputs)
                pred_list[i] = torch.cat((pred_list[i], feature), 0)
                prob_list[i] = torch.cat((prob_list[i], prob), 0)
        
#        for i in range(5):
#            out = pred_list[i].cpu().numpy()
#            df = pd.DataFrame(out)
#            df['filename'] = filename
#            df.to_csv('lgb/DL_feature/' + 'resnet34_' + str(i) + '_test.csv', index=None)
#    
#            out = prob_list[i].cpu().numpy()
#            df = pd.DataFrame(out)
#            df['filename'] = filename
#            df.to_csv('lgb/DL_feature/prob_' + 'resnet34_' + str(i) + '.csv', index=None)
            
        for i in range(5):
            out = pred_list[i].cpu().numpy()
            df = pd.DataFrame(out)
            df['filename'] = filename
            df.to_csv('lgb/DL_feature2/' + 'myecgnet_' + str(i) + '_test.csv', index=None)
    
            out = prob_list[i].cpu().numpy()
            df = pd.DataFrame(out)
            df['filename'] = filename
            df.to_csv('lgb/DL_feature2/prob_' + 'myecgnet_' + str(i) + '.csv', index=None)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
#    parser.add_argument("command", metavar="<command>", help="train or infer")
    parser.add_argument("--command", type=str, help="train or infer", default='test')
#     parser.add_argument("--ckpt", type=str, help="the path of model weight file")
#     parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--fold", type=int, help="fold")
    parser.add_argument("--test_dir", type=str, help="dir")
    parser.add_argument("--test_label", type=str, help="label")
    parser.add_argument("--resume", type=bool, default=False)
    args = parser.parse_args()
    
#    args.test_label='hf_round2_train.txt'
#    args.test_dir='/home/hcb/桌面/ecg_pytorch-master/hf_round2_train'
#    args.fold=0
    print(args.command)
#    args.command = 'train'
#    args.model_name = 'resnet34'
    # args.ckpt = 'ckpt/myCNN/'
#    args.resume = True
    print(args.resume)
    for i in range(5):
        #if i != 0:
        #    continue
        config.train_data = 'path/train'
        args.fold = i
        if (args.command == "train"):  
            train(args)
        if (args.command == "val"):
            val(args)
    if (args.command == 'get_feature'):
        get_feature(args)
    if (args.command == "test"):
        args.fold = 0
        test(args)

