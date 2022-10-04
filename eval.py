import argparse
import datetime
import os.path as osp
import time

import torch
import torch.utils.data

from datasets import build_test_loader, build_train_loader
from defaults import get_default_cfg
from engine import evaluate_performance, train_one_epoch
from models.seqnet import SeqNet
from utils.utils import mkdir, resume_from_ckpt, save_on_master, set_random_seed


def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Creating model")
    model = SeqNet(cfg)
    model.to(device)

    print("Loading data")
    train_loader = build_train_loader(cfg)
    gallery_loader, query_loader = build_test_loader(cfg)


    start_epoch = args.start_epoch


    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
   
    print("Start Evaluation")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        args.ckpt = osp.join(output_dir, f"epoch_{epoch}.pth")

        resume_from_ckpt(args.ckpt, model)
        evaluate_performance(
            model,epoch,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
        )
            
        
        # evaluate_performance(
        #     model,
        #     gallery_loader,
        #     query_loader,
        #     device,
        #     use_gt=cfg.EVAL_USE_GT,
        #     use_cache=cfg.EVAL_USE_CACHE,
        #     use_cbgm=cfg.EVAL_USE_CBGM,
        # )

        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total Evaluation time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg",  default='./configs/prw.yaml', dest="cfg_file", help="Path to configuration file.")
    parser.add_argument(
        "--eval", default=True, action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument(
        "--resume", default =True, action="store_true", help="Resume from the specified checkpoint."
    )
    parser.add_argument(
        "--start_epoch", default =10, action="store_true", help="Resume from the specified checkpoint."
    )
    parser.add_argument("--ckpt", default='./exp_prw/epoch_10.pth', help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    args = parser.parse_args()
    main(args)
