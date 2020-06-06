# python test.py
#python main.py --test_dir /home/hcb/桌面/ecg_pytorch-master/hf_round2_train --command test --test_label /home/hcb/桌面/ecg_pytorch-master/hf_round2_train.txt
#python main.py --test_dir /tcdata/hf_round2_testA --command test --test_label /tcdata/hf_round2_subA.txt
#python main.py --command get_feature --model_name myecgnet
#python main2.py --command get_feature --model_name myecgnet
cd lgb
#python get_mlfeature.py
python get_test.py
