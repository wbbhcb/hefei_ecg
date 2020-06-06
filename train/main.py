# -*- coding: utf-8 -*-
'''
@time: 2019/7/23 19:42

@ author: javis
'''
import torch, time, os, shutil
import models, utils, utils2
import numpy as np
import pandas as pd
from tensorboard_logger import Logger
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import ECGDataset, my_collate_fn
from config2 import config
import tqdm
from radam import RAdam
from label import get_label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(41)
torch.cuda.manual_seed(41)


# 保存当前模型的权重，并且更新最佳的模型权重
def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, config.current_w)
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)


def train_epoch(model, optimizer, criterion, train_dataloader, epoch, lr, best_f1, 
                val_dataloader, model_save_dir, state, round_):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    #tq = tqdm.tqdm(total=len(train_dataloader)*config.batch_size)
    #tq.set_description('epoch %d, lr %.4f, best_f:%.4f' % (epoch, lr, best_f1))

    for i, (inputs, target) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output, _ = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        if args.model_kind == 1:
            f1 = utils.calc_f1(target, output, 0.5)
        else:
            f1 = utils2.calc_f1(target, output, 0.5)
        f1_meter += f1
        #tq.update(config.batch_size)
         
        if epoch > round_ and i % 127 == 126:
            val_loss, val_f1, _, _ = val_epoch(model, criterion, val_dataloader)
            if best_f1 < val_f1:
                best_f1 = val_f1
                state['state_dict']=model.state_dict()
                save_ckpt(state, True, model_save_dir)
#                print('save best')
            print('#epoch:%02d  val_loss:%0.3e val_f1:%.3f'
                  % (epoch, val_loss, val_f1))
        
        #tq.set_postfix(loss="%.4f   f1:%.3f" % (loss.item(), f1))
    #tq.close()
        #if it_count != 0 and it_count % show_interval == 0:
#        print("%d,loss:%.3e f1:%.3f" % (it_count, loss.item(), f1), end='\r')
    return loss_meter / it_count, f1_meter / it_count, best_f1


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
#        tq = tqdm.tqdm(total=len(val_dataloader) * config.batch_size)
        for inputs, target in val_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output, _ = model(inputs)
            it_count += 1
            label_all = torch.cat((label_all, target), 0)
            pred_all = torch.cat((pred_all, output), 0)
#            tq.update(config.batch_size)
#        tq.close()
        output = pred_all
        target = label_all
        loss = criterion(output, target)
        loss_meter = loss.item()

        output = torch.sigmoid(output)
        if args.model_kind == 1:
            f1 = utils.calc_f1(target, output, threshold)
        else:
            f1 = utils2.calc_f1(target, output, threshold)
        acc, true_positives, real_positives, predicted_positives = utils.calc_acc_f1(target, output, threshold)
        
        fout = open('log.txt', 'a+', encoding='utf-8')
        fout.write('\n' + '*'*20 + '\n')
        fout.write('acc:' + str(acc) + '\n')
        fout.write('true_positives:' + str(true_positives) + '\n')
        fout.write('real_positives:' + str(real_positives) + '\n')
        fout.write('predicted_positives:' + str(predicted_positives) + '\n')
        fout.close()
        
        # f1_meter += f1
    return loss_meter, f1, target, output


def train(args):
    # model
    print(args.model_name)
    config.train_data = config.train_data + str(args.fold) + '.pth'
#    config.train_data = config.train_data + 'trainsfer_' + str(args.fold) + '.pth'
    
    config.model_name = args.model_name
    model = getattr(models, config.model_name)()

    model = model.to(device)
    # data
    if args.model_kind == 1:
        import dataset2
        train_dataset = dataset2.ECGDataset(data_path=config.train_data, train=True, transform=True)
        train_dataloader = DataLoader(train_dataset, collate_fn=my_collate_fn,
                                  batch_size=config.batch_size, shuffle=True, num_workers=6)
    else:
        train_dataset = ECGDataset(data_path=config.train_data, train=True, transform=True)
        train_dataloader = DataLoader(train_dataset, #collate_fn=my_collate_fn,
                                      batch_size=config.batch_size, shuffle=True, num_workers=6)
    
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=6)
    print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=config.lr)
    w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
    if args.model_kind == 1:
        criterion = utils.WeightedMultilabel(w)
        print(1)
    else:
        criterion = utils2.WeightedMultilabel(w)
#    criterion = utils.My_loss(w)
    # 模型保存文件夹
    model_save_dir = '%s/%s' % (config.ckpt + str(args.model_kind), 
                                config.model_name + '_' + str(args.fold))
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
    else:
        path ='%s/%s' % (config.ckpt, config.model_name + '_transfer')
        print(path)
        current_w = torch.load(os.path.join(path, config.best_w))
        model.load_state_dict(current_w['state_dict'])
            
    logger = Logger(logdir=model_save_dir, flush_secs=2)
    # =========>开始训练<=========
    val_loss = 10
    val_f1 = -1
    state = {}
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_f1, best_f1 = train_epoch(model, optimizer, criterion, train_dataloader, epoch, lr, best_f1, 
                                   val_dataloader, model_save_dir, state, 0)
        # if epoch % 2 == 1:
        val_loss, val_f1, _, _ = val_epoch(model, criterion, val_dataloader)
        print('#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s'
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
        else:
            save_ckpt(state, False, model_save_dir)
        best_f1 = max(best_f1, val_f1)

        if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay
#            best_w = os.path.join(model_save_dir, config.best_w)
#            model.load_state_dict(torch.load(best_w)['state_dict'])
            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)


def transfer_train(args):
    print(args.model_name)
    config.train_data = config.train_data + 'trainsfer.pth'
    config.model_name = args.model_name
    model = getattr(models, config.model_name)()
    model = model.to(device)
    import dataset2
    train_dataset = dataset2.ECGDataset(data_path=config.train_data, train=True, transfer=True,
                               transform=True)
    train_dataloader = DataLoader(train_dataset, collate_fn=my_collate_fn,
                                  batch_size=config.batch_size, shuffle=True, num_workers=6)
    val_dataset = ECGDataset(data_path=config.train_data, train=False, transfer=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=6)
    print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=config.lr)
    w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
    criterion = utils.WeightedMultilabel2(w)
#    criterion = utils.My_loss(w)
    # 模型保存文件夹
    model_save_dir = '%s/%s' % (config.ckpt, config.model_name + '_transfer')
    args.ckpt = model_save_dir
    # if args.ex: model_save_dir += args.ex
    best_f1 = -1
    lr = 3e-4
    start_epoch = 1
    stage = 1
    # 从上一个断点，继续训练
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
        
    if args.resume:
        if os.path.exists(args.ckpt):  # 这里是存放权重的目录
            # model_save_dir = args.ckpt
            current_w = torch.load(os.path.join(args.ckpt, config.best_w))
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
    # =========>开始训练<=========
    val_loss = 10
    val_f1 = -1
    state = {}
    for epoch in range(start_epoch, 25 + 1):
        since = time.time()
        train_loss, train_f1, best_f1 = train_epoch(model, optimizer, criterion, train_dataloader, epoch, lr, best_f1, 
                                           val_dataloader, model_save_dir, state, 50)
        # if epoch % 2 == 1:
        val_loss, val_f1, _, _ = val_epoch(model, criterion, val_dataloader)
        print('#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s'
              % (epoch, stage, train_loss, train_f1, val_loss, val_f1, utils.print_time_cost(since)))
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,
                 'stage': stage, "best_f": val_f1}
        if best_f1 < val_f1:
            save_ckpt(state, best_f1 < val_f1, model_save_dir)
            print('save best')
        else:
            save_ckpt(state, False, model_save_dir)
        best_f1 = max(best_f1, val_f1)

        if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay
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
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=6)
    for threshold in list_threhold:
        val_loss, val_f1, target, output = val_epoch(model, criterion, val_dataloader, threshold)
        print('threshold %.2f val_loss:%0.3e val_f1:%.4f\n' % (threshold, val_loss, val_f1))
    return target, output

#提交结果使用
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
    for fold in range(1):
        model_save_dir = '%s/%s' % (config.ckpt, config.model_name + '_' + str(fold))
        print(model_save_dir)
        model = getattr(models, config.model_name)()
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
        
    out = out #/ 5
    num = 0
    for line in open(config.test_label, encoding='utf-8'):
        fout.write(line.strip('\n'))
        output = out[num]
        ixs = [i for i, out in enumerate(output) if out > 0.5]
        for i in ixs:
            fout.write("\t" + idx2name[i])
        fout.write('\n')
        num = num + 1
    fout.close()
        
    
def check(args):
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
    dd = torch.load(config.train_data)
    filename = dd['train']
    val_dataset = ECGDataset(data_path=config.train_data, train=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, 
                                shuffle=False, num_workers=6)
    
    for threshold in list_threhold:
        val_loss, val_f1, target, output = val_epoch(model, criterion, val_dataloader, threshold)
        print('threshold %.2f val_loss:%0.3e val_f1:%.3f\n' % (threshold, val_loss, val_f1))
    
    return target, output, filename


def get_feature(args):
    from dataset import ECGDataset_test
    config.model_name = args.model_name
    print(config.model_name)
    model_save_dir = '%s/%s' % (config.ckpt + str(args.model_kind), 
                                config.model_name + '_' + str(args.fold))
#    utils.mkdirs(config.sub_dir)
    # model
    model = getattr(models, config.model_name)()
    model.load_state_dict(torch.load(os.path.join(model_save_dir, config.best_w), 
                                     map_location='cpu')['state_dict'])
    model = model.to(device)
    model.eval()
    config.test_label = 'hf_round2_train.txt'
    config.test_dir = config.train_dir
    path_all = []
    prob_all = []
    filename = []
    with torch.no_grad():
        for line in open(config.test_label, encoding='utf-8'):
#            fout.write(line.strip('\n'))
            id = line.split('\t')[0]
            file_path = os.path.join(config.test_dir, id)
            filename.append(id)
            path_all.append(file_path)
        test_dataset = ECGDataset_test(path_all)
        test_load = DataLoader(test_dataset, batch_size=64, num_workers=6, 
                               shuffle=False)
        
        if torch.cuda.is_available():
            pred_all = torch.Tensor().cuda()
            prob_all = torch.Tensor().cuda()
        else:
            pred_all = torch.Tensor()
            prob_all = torch.Tensor()
        for inputs, target in tqdm.tqdm(test_load):
            inputs = inputs.to(device)
            prob, feature = model(inputs)
#            print(feature.shape)
#            output = torch.sigmoid(out)
            pred_all = torch.cat((pred_all, feature), 0)
            prob_all = torch.cat((prob_all, prob), 0)
        out = pred_all.cpu().numpy()
        df = pd.DataFrame(out)
        df['filename'] = filename
        df.to_csv('lgb/DL_feature'+ str(args.model_kind) +'/' + args.model_name + '_' + str(args.fold) + '_test.csv', index=None)


        out = prob_all.cpu().numpy()                
        df = pd.DataFrame(out)
        df['filename'] = filename
        
        df.to_csv('lgb/DL_feature'+ str(args.model_kind)+'/prob_' + args.model_name + '_' + str(args.fold) + '.csv', index=None)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
#    parser.add_argument("command", metavar="<command>", help="train or infer")
    parser.add_argument("--command", type=str, help="train or infer", default='train')
#     parser.add_argument("--ckpt", type=str, help="the path of model weight file")
#     parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--fold", type=int, help="fold")
    parser.add_argument("--test_dir", type=int, help="dir")
    parser.add_argument("--test_label", type=int, help="label")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--model_kind", type=int)
    args = parser.parse_args()
    
#    args.test_label='hf_round2_train.txt'
#    args.test_dir='/home/hcb/桌面/ecg_pytorch-master/hf_round2_train'
#    args.fold=0
    
#    args.command = 'train'
#    args.model_name = 'resnet34'
    # args.ckpt = 'ckpt/myCNN/'
#    args.resume = True
    print(args.resume)
    for i in range(5):
#        if i not in [0]:
#            continue
        config.train_data = 'path/train'
        args.fold = i
        print(i)
        if (args.command == "train"):  
            train(args)
        #break    
        if (args.command == 'get_feature'):
            get_feature(args)
#        
#        if (args.command == 'check'):
#            check(args)
    if (args.command == "val"):
        if torch.cuda.is_available():
            label_all = torch.Tensor().cuda()
            pred_all = torch.Tensor().cuda()
        else:
            label_all = torch.Tensor()
            pred_all = torch.Tensor()
        for i in range(5):
            #if i!=3:
            #    continue
            config.train_data = 'path/train'
            args.fold = i
            target, output = val(args)
            label_all = torch.cat((label_all, target), 0)
            pred_all = torch.cat((pred_all, output), 0)
        f1 = utils.calc_f1(label_all, pred_all, 0.5)
        acc, true_positives, real_positives, predicted_positives = utils.calc_acc_f1(label_all, pred_all, 0.5)
        
        fout = open('log.txt', 'a+', encoding='utf-8')
        fout.write('\n' + '*'*20 + '\n')
        fout.write('acc:' + str(acc) + '\n')
        fout.write('true_positives:' + str(true_positives) + '\n')
        fout.write('real_positives:' + str(real_positives) + '\n')
        fout.write('predicted_positives:' + str(predicted_positives) + '\n')
        fout.close()		
#        acc, true_positives, real_positives, predicted_positives = utils.calc_acc_f1(target, output, 0.5)
        print('f1:%.4f' %(f1))
        
    if (args.command == "check"):
        
        filenameslist, filelabelslist = get_label()
        pred = np.zeros(filelabelslist.shape)
        if torch.cuda.is_available():
            label_all = torch.Tensor().cuda()
            pred_all = torch.Tensor().cuda()
        else:
            label_all = torch.Tensor()
            pred_all = torch.Tensor()
        for i in range(5):
            config.train_data = 'path/train'
            args.fold = i
            target, output, filename = check(args)
            idx = []
            for tmp_name in filename:
                idx.append(filenameslist.index(tmp_name))
            pred[idx] += np.round(output.cpu().detach().numpy())
            label_all = torch.cat((label_all, target), 0)
            pred_all = torch.cat((pred_all, output), 0)
        f1 = utils.calc_f1(label_all, pred_all, 0.5)
        acc, true_positives, real_positives, predicted_positives = utils.calc_acc_f1(label_all, pred_all, 0.5)
        
        fout = open('log.txt', 'a+', encoding='utf-8')
        fout.write('\n' + '*'*20 + '\n')
        fout.write('acc:' + str(acc) + '\n')
        fout.write('true_positives:' + str(true_positives) + '\n')
        fout.write('real_positives:' + str(real_positives) + '\n')
        fout.write('predicted_positives:' + str(predicted_positives) + '\n')
        fout.close()		
#        acc, true_positives, real_positives, predicted_positives = utils.calc_acc_f1(target, output, 0.5)
        print('f1:%.4f' %(f1))
        df = pd.DataFrame(abs(pred-4*filelabelslist))
        df['filename'] = filenameslist
        df.to_csv('label_eda.csv', index=None)
    if (args.command == "test"):
        args.fold = 0
        test(args)

    if (args.command == 'transfer'):
        transfer_train(args)
