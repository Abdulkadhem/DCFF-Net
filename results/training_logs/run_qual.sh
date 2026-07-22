cd /root/gf/code
export HF_HOME=/root/gf/hf HF_HUB_DISABLE_SYMLINKS_WARNING=1
/venv/main/bin/python make_qualitative.py --ckpt results/dcff_final_best.pt --n 6 --out /root/gf/qualitative.png
