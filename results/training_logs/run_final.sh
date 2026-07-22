export HF_HOME=/root/gf/hf
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd /root/gf/code
/venv/main/bin/python train.py --model dcff --backbone resnet18 --pos_weight 2 --epochs 100 --no_cbam --no_boundary --tag dcff_final
echo FINAL_DONE
