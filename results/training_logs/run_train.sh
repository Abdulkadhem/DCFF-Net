export HF_HOME=/root/gf/hf
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd /root/gf/code
echo "=== DCFF-Net ==="
/venv/main/bin/python train.py --model dcff --epochs 100 --batch 16 --tag dcff --workers 4
echo "=== FC-Siam baseline ==="
/venv/main/bin/python train.py --model fcsiam --epochs 100 --batch 16 --tag fcsiam --workers 4
echo ALL_TRAINING_DONE
