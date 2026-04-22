#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from common import ensure_exists, require_packages, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ACT policy on a local Realman LeRobot dataset.")
    parser.add_argument("--dataset", "-d", required=True, help="Local LeRobot dataset directory.")
    parser.add_argument("--repo-id", required=True, help="Local repo id used when opening the dataset.")
    parser.add_argument("--output", "-o", default="outputs/act_realman", help="Checkpoint output directory.")
    parser.add_argument("--device", default="cuda", help="cuda, cpu or mps.")

    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--steps", type=int, default=20000, help="Number of gradient steps.")
    parser.add_argument("--log-freq", type=int, default=100, help="How often to print scalar summaries.")
    parser.add_argument("--save-freq", type=int, default=5000, help="Checkpoint frequency.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")

    parser.add_argument("--chunk-size", type=int, default=64, help="ACT action chunk size.")
    parser.add_argument("--n-action-steps", type=int, default=64, help="Action steps executed per policy query.")
    parser.add_argument("--vision-backbone", default="resnet18", help="torchvision resnet backbone.")
    parser.add_argument("--dim-model", type=int, default=512, help="Transformer model dimension.")
    parser.add_argument("--n-heads", type=int, default=8, help="Transformer attention heads.")
    parser.add_argument("--dim-feedforward", type=int, default=3200, help="Transformer feedforward dim.")
    parser.add_argument("--n-encoder-layers", type=int, default=4, help="Transformer encoder layers.")
    parser.add_argument("--n-decoder-layers", type=int, default=1, help="Transformer decoder layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    parser.add_argument("--latent-dim", type=int, default=32, help="ACT VAE latent dimension.")
    parser.add_argument("--n-vae-encoder-layers", type=int, default=4, help="ACT VAE encoder layers.")
    parser.add_argument("--kl-weight", type=float, default=10.0, help="ACT KL loss weight.")
    parser.add_argument("--optimizer-lr", type=float, default=1e-5, help="Optimizer learning rate.")
    parser.add_argument(
        "--optimizer-weight-decay",
        type=float,
        default=1e-4,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--optimizer-lr-backbone",
        type=float,
        default=1e-5,
        help="Backbone learning rate stored in the config.",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="Optional temporal ensemble coefficient for deployment.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="realman-act", help="WandB project name.")
    parser.add_argument("--wandb-run", default=None, help="Optional WandB run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_packages(["lerobot"])

    from lerobot.configs.types import FeatureType
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    dataset_path = ensure_exists(args.dataset, "LeRobot dataset")
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )

    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    dataset_meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=dataset_path)
    policy_features = dataset_to_policy_features(dataset_meta.features)
    input_features = {key: value for key, value in policy_features.items() if value.type is not FeatureType.ACTION}
    output_features = {key: value for key, value in policy_features.items() if value.type is FeatureType.ACTION}

    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        vision_backbone=args.vision_backbone,
        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        latent_dim=args.latent_dim,
        n_vae_encoder_layers=args.n_vae_encoder_layers,
        dropout=args.dropout,
        kl_weight=args.kl_weight,
        optimizer_lr=args.optimizer_lr,
        optimizer_weight_decay=args.optimizer_weight_decay,
        optimizer_lr_backbone=args.optimizer_lr_backbone,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        device=args.device,
    )

    policy = ACTPolicy(config)
    policy.train()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=dataset_meta.stats)
    fps = dataset_meta.fps
    delta_timestamps = {"action": [step / fps for step in range(args.chunk_size)]}
    dataset = LeRobotDataset(repo_id=args.repo_id, root=dataset_path, delta_timestamps=delta_timestamps)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    optimizer = config.get_optimizer_preset().build(policy.parameters())

    run_manifest = {
        "dataset": str(dataset_path),
        "repo_id": args.repo_id,
        "frames": len(dataset),
        "episodes": dataset.num_episodes,
        "fps": dataset.fps,
        "input_features": list(input_features.keys()),
        "output_features": list(output_features.keys()),
        "config": vars(args),
    }
    save_json(output_dir / "train_run.json", run_manifest)

    print("=" * 72)
    print("Realman ACT training")
    print("=" * 72)
    print(f"dataset         : {dataset_path}")
    print(f"repo_id         : {args.repo_id}")
    print(f"output          : {output_dir}")
    print(f"device          : {device}")
    print(f"frames          : {len(dataset)}")
    print(f"episodes        : {dataset.num_episodes}")
    print(f"fps             : {dataset.fps}")
    print(f"input_features  : {list(input_features.keys())}")
    print(f"output_features : {list(output_features.keys())}")
    print("=" * 72)

    step = 0
    start_time = time.time()
    running_loss = 0.0
    progress = tqdm(total=args.steps, desc="train", dynamic_ncols=True, unit="step")

    while step < args.steps:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item())
            step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if args.wandb:
                log_payload = {"loss": float(loss.item()), "step": step}
                for key, value in loss_dict.items():
                    log_payload[key] = float(value)
                wandb.log(log_payload, step=step)

            if step % args.log_freq == 0:
                avg_loss = running_loss / args.log_freq
                running_loss = 0.0
                parts = [f"step={step}", f"loss={avg_loss:.4f}"]
                for key in ("l1_loss", "kld_loss"):
                    if key in loss_dict:
                        parts.append(f"{key}={float(loss_dict[key]):.4f}")
                print(" | ".join(parts))

            if step % args.save_freq == 0:
                checkpoint_dir = output_dir / f"checkpoint_{step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                print(f"saved checkpoint: {checkpoint_dir}")

            if step >= args.steps:
                break

    progress.close()

    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)

    elapsed = time.time() - start_time
    print("\nTraining finished.")
    print(f"elapsed_min      : {elapsed / 60:.1f}")
    print(f"final_checkpoint : {output_dir}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
