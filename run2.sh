# train
#cp -r config2/config2.py train/config2.py
#cp -r config2/config_lgb.py train/lgb/config_lgb.py
cd train
#python data_process.py --kind first
#python data_process.py --kind third

#python main.py --model_name myecgnet --command transfer
#python main.py --model_name myecgnet --command train --model_kind 1
#python main.py --model_name myecgnet --command train --model_kind 2
#python main.py --model_name myecgnet --command get_feature --model_kind 1
#python main.py --model_name myecgnet --command get_feature --model_kind 2
cd lgb
#python get_mlfeature.py
#python train.py
#python train2.py
cd ..
cd ..
# predict
cd classifier
python main.py --command get_feature --model_name myecgnet
python main2.py --command get_feature --model_name myecgnet
cd lgb
python get_mlfeature.py
python get_test.py

