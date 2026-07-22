export HF_HOME=/root/gf/hf
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd /root/gf/code
echo "== dcff resnet34 =="
/venv/main/bin/python train.py --model dcff --backbone resnet34 --pos_weight 2 --epochs 120 --tag dcff_r34
echo ALL_TRAINING_DONE
