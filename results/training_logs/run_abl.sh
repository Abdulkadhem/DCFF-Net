export HF_HOME=/root/gf/hf
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd /root/gf/code
/venv/main/bin/python train.py --model dcff --pos_weight 2 --epochs 60 --fusion dual --tag abl_dual
/venv/main/bin/python train.py --model dcff --pos_weight 2 --epochs 60 --fusion diff --tag abl_diff
/venv/main/bin/python train.py --model dcff --pos_weight 2 --epochs 60 --fusion conc --tag abl_conc
/venv/main/bin/python train.py --model dcff --pos_weight 2 --epochs 60 --no_cbam --tag abl_nocbam
/venv/main/bin/python train.py --model dcff --pos_weight 2 --epochs 60 --no_aspp --tag abl_noaspp
/venv/main/bin/python train.py --model dcff --pos_weight 2 --epochs 60 --no_boundary --tag abl_nobnd
/venv/main/bin/python train.py --model dcff --pos_weight 2 --epochs 60 --pretrained 0 --tag abl_scratch
echo ALL_ABLATION_DONE
