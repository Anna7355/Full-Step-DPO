import os
import torch
import wandb
import datasets
import json
import transformers
from omegaconf import OmegaConf
from accelerate import Accelerator
from trainer import Trainer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from models import Union_Model
from utils import *
from collators import Align_Collator
from StepRunners import AlignRunner
import argparse
import bitsandbytes as bnb
from datasets import load_dataset


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(align_config_path):
    config = OmegaConf.load(align_config_path)
    setup_seed(config.seed)
    accelerator = Accelerator(mixed_precision=config.mixed_precision,
                            gradient_accumulation_steps=config.gradient_accumulation, cpu=False)

    if accelerator.is_local_main_process:
        wandb.init(project=config.project_name, name=config.run_name)

    policy_model = transformers.AutoModelForCausalLM.from_pretrained(config.policy_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    ref_model = transformers.AutoModelForCausalLM.from_pretrained(config.ref_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    model_setting(policy_model,config.use_grad_ckpt)
    model_setting(ref_model,config.use_grad_ckpt)
    policy_tokenizer = transformers.AutoTokenizer.from_pretrained(config.policy_tokenizer_name)
    policy_tokenizer.pad_token_id = policy_tokenizer.eos_token_id

    union_model = Union_Model(policy_model, policy_tokenizer, accelerator,ref_model=ref_model)
    if accelerator.is_local_main_process:
        wandb.watch(union_model)

    if config.data_path == "xinlai/Math-Step-DPO-10K":
        ds = load_dataset(config.data_path, split='train')
        train_set = convert_step_data(ds)
    else:
        with open(config.data_path,"r") as f:
            train_set = json.load(f)
    
    train_set = datasets.Dataset.from_list(train_set) 
    collator = Align_Collator(policy_tokenizer, config)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, collate_fn=collator.collate_fn,shuffle=True,drop_last=True)

    Trainer.StepRunner = AlignRunner
    Trainer.save_ckpt = AlignRunner.save_ckpt

    # optimizer_class = getattr(torch.optim, config.optimizer)
    optimizer_class = bnb.optim.Adam8bit
    optimizer = optimizer_class(union_model.parameters(), lr=config.learning_rate)


    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (config.warmup_steps + 1)))
    trainer = Trainer(union_model, accelerator = accelerator,
                            optimizer=optimizer,tokenizer=policy_tokenizer, lr_scheduler=lr_scheduler, config = config)

    trainer.fit(dataloader=train_loader,
                    epochs=config.epoch,ckpt_path = config.ckpt_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train alignment model.')
    parser.add_argument('--align_config', type=str, default="./configs/qwen_align_config.yaml",
                        help='Path to the configuration file')
    
    args = parser.parse_args()
    main(args.align_config)
