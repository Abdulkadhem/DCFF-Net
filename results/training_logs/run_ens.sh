cd /root/gf/code
export HF_HOME=/root/gf/hf
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
/venv/main/bin/python ensemble_eval.py   --model "/root/gf/code/results/dcff_final_best.pt,resnet18,0,1"   --model "/root/gf/code/F:/2026projects/ConformalCD/results/dcff_r34_best.pt,resnet34,1,1"   --model "/root/gf/code/F:/2026projects/ConformalCD/results/dcff_pw1_best.pt,resnet18,1,1"
