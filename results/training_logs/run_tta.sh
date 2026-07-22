cd /root/gf/code
export HF_HOME=/root/gf/hf HF_HUB_DISABLE_SYMLINKS_WARNING=1
/venv/main/bin/python eval_tta.py --ckpt "/root/gf/code/results/dcff_final_best.pt" --backbone resnet18 --no_cbam
