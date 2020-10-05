Source code for paper "Protect Privacy of Deep Classification Networks by Exploiting
Their Generative Power"

## Environment
Python=3.7 \
Pytorch=1.3.1

## Train models at different phases
### Repo structure:
- Detailed parameters are included in params/*.txt
- Save your models as following:
   - Save your models under: models/dataset_name/your_model_name
   - Save your meminf data under: meminf_data/dataset_name/(auto_generated)
   - Save your attack models under: attack_models/dataset_name/(auto_generated)
### Train a classifier
```
python train_jem.py @params/cls_params.txt
```
### Train a JEM from scratch
```
python train_jem.py @params/jem_params.txt
```
### Transferring to JEM
```
python train_jem.py @params/transfer_params.txt
```
### Sample from a JEM
```
python train_jem.py @params/sample_params.txt
```
### Retrain/Fine-tune the model
```
python train_jem.py @params/retrain_params.txt
```
### Classifier(and JEM's classifier) evaluation
```
python eval_wrn_ebm.py --load_path=YOUR_MODEL --eval=test_clf --dataset=cifar_test
```

## Shadow model attack
### Generate (label, output, in_out) pairs
```
python gen_meminf_data.py --load_from=YOUR_SHADOW_MODEL
```
  
### Train attack models
```
python train_attack_model.py --target_class=0 --path_prefix=meminf_data/cifar10/YOUR_SHADOW_MODEL
```

### Evaluate attack models
```
python eval_attack_model.py --all_classes --att_model_prefix=attack_models/cifar10/YOUR_SHADOW_MODEL --eval
```

## Train classifier with defenses
### L2/dropout
- Set weight_decay and dropout in parameters
### DP-SGD
```
python train_jem.py @params/dp_params.txt
```
### Min-Max
```
python train_jem.py @params/min_max_params.txt
```

