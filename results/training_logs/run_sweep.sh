export HF_HOME=/root/gf/hf
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd /root/gf/code
echo "== dcff pw2 =="
/venv/main/bin/python train.py --model dcff --pos_weight 2 --epochs 80 --tag dcff_pw2
echo "== dcff pw1 =="
/venv/main/bin/python train.py --model dcff --pos_weight 1 --epochs 80 --tag dcff_pw1
echo "== fcsiam =="
/venv/main/bin/python train.py --model fcsiam --pos_weight 2 --epochs 80 --tag fcsiam
echo ALL_TRAINING_DONE
