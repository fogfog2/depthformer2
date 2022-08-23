export PYTHONPATH="${PYTHONPATH}:/home/sj/src/depthformer2" 


export WANDB_ENTITY="fogfog2"
export WANDB_PROJECT="overfit_custom_cmt_H"
python scripts/train.py /home/sj/src/depthformer2/configs/overfit_custom_cmt_H.yaml

export WANDB_PROJECT="overfit_custom_cmt_H_attention"
python scripts/train.py /home/sj/src/depthformer2/configs/overfit_custom_cmt_H_attention.yaml


export WANDB_PROJECT="overfit_custom_resnet_fbnet"
python scripts/train_fbnet.py /home/sj/src/depthformer2/configs/overfit_custom_resnet_fbnet.yaml

export WANDB_PROJECT="overfit_custom_cmt_H_fbnet"
python scripts/train_fbnet.py /home/sj/src/depthformer2/configs/overfit_custom_cmt_H_fbnet.yaml

export WANDB_PROJECT="overfit_custom_cmt_H_attention_fbnet"
python scripts/train_fbnet.py /home/sj/src/depthformer2/configs/overfit_custom_cmt_H_attention_fbnet.yaml



