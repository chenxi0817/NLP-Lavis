"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *

# train.py 顶部
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    # =================================================================
    # --- 新增的LoRA/QLoRA改造逻辑 ---
    # print("======> 正在对LLM应用LoRA/QLoRA改造 <======")
    
    # # 步骤A（QLoRA需要）：为k-bit量化训练做准备
    # # 这会修复一些兼容性问题
    # model.opt_model = prepare_model_for_kbit_training(model.opt_model)
    
    # # 步骤B：定义LoRA配置
    # lora_config = LoraConfig(
    #     r=16,  # LoRA的秩，一个关键超参数，通常是8, 16, 32, 64
    #     lora_alpha=32, # LoRA的alpha，通常是r的两倍
    #     target_modules=["q_proj", "v_proj"], # 指定要对哪些线性层应用LoRA，对于OPT模型，通常是q_proj和v_proj
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM" # 任务类型，对于OPT是因果语言模型
    # )
    
    # # 步骤C：将LoRA配置应用到模型上
    # model.opt_model = get_peft_model(model.opt_model, lora_config)


    trainable_params = 0
    for p in model.parameters():
        if p.requires_grad:
            print(f"  - {p.shape} ({p.numel()})")
            trainable_params += p.numel()
    print(f"可训练参数总量: {trainable_params:,}")
    # 步骤D（可选但推荐）：打印出可训练参数，验证LoRA是否应用成功
    # if hasattr(model.opt_model, "print_trainable_parameters"):
    #     # 如果opt_model有这个方法，直接调用
    #     model.opt_model.print_trainable_parameters()
    # else:
    #     # 否则手动计算并打印
    #     total_params = sum(p.numel() for p in model.parameters())
    #     for p in model.parameters():
    #         if p.requires_grad:
    #             print(f"  - {p.shape} ({p.numel()})")
    #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    #     print(f"可训练参数总量: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% 的总参数)")
    #     print(f"总参数量: {total_params:,}")
        
    #     # 打印LoRA层
    #     print("LoRA层:")
    #     for name, param in model.named_parameters():
    #         if "lora" in name.lower() and param.requires_grad:
    #             print(f"  - {name} ({param.shape})")
    # print("======> LoRA/QLoRA 改造完成 <======")
    # =================================================================


    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
