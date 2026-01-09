# Full-Step-DPO

Repo for "[Full-Step-DPO: Self-Supervised Preference Optimization with Step-wise Rewards for Mathematical Reasoning](https://aclanthology.org/2025.findings-acl.1249/)".


### Environment Setup

---

```bash
conda create -n fsdpo python=3.10 -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```


### Model Training

---

```bash
accelerate launch --config_file configs/accelerate_config.yaml train_align.py --align_config ./configs/qwen_align_config.yaml
```

### Evaluation

---

```bash
python eval.py \
    --model  \
    --tokenizer_name  \
    --data_file ./data/test/GSM8K_test_data.jsonl \
    --save_path "eval_results/gsm8k.json" \
    --prompt "alpaca-cot-step" \
    --tensor_parallel_size 1 \
    --verifier "default" \
    --temp 0 \
    --sampling_size 1
```

### Acknowledgement

---

This repository is based on [Step-DPO](https://github.com/dvlab-research/Step-DPO).

Many thanks for their efforts!


### Citation

---

If you find this project useful in your research, please consider citing us:

```bash
@article{xu2025full,
  title={Full-step-dpo: Self-supervised preference optimization with step-wise rewards for mathematical reasoning},
  author={Xu, Huimin and Mao, Xin and Li, Feng-Lin and Wu, Xiaobao and Chen, Wang and Zhang, Wei and Luu, Anh Tuan},
  journal={arXiv preprint arXiv:2502.14356},
  year={2025}
}
```