import os
import sys
import argparse
import logging
import math
import signal
import time

from datetime import datetime, timedelta
from types import FrameType
from typing import Any

from config import load_classifier_config

DEFAULT_CONFIG_CLI_PATH = "open_world_filter/configs/default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Fish Classification Model")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_CLI_PATH,
        help="Path to the coarse classifier YAML config file",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return resolve_args(build_parser().parse_args())


def resolve_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    config = load_classifier_config(cli_args.config)
    values = {
        "config": cli_args.config,
        "train_dir": config.data.train_dir,
        "val_dir": config.data.val_dir,
        "batch_size": config.training.batch_size,
        "epochs": config.training.epochs,
        "lr": config.training.lr,
        "weight_decay": config.training.weight_decay,
        "image_size": config.training.image_size,
        "model_path": config.training.model_path,
        "save_path": config.outputs.save_path,
        "save_frequency": config.training.save_frequency,
        "num_workers": config.training.num_workers,
        "backbone": config.training.backbone,
        "use_class_pairs": config.training.use_class_pairs,
        "profile_batches": config.training.profile_batches,
        "gradient_clip": config.training.gradient_clip,
        "warmup_epochs": config.training.warmup_epochs,
        "drop_last": config.training.drop_last,
        "samples_per_class": config.training.samples_per_class,
        "local_rank": -1,
        "val_frequency": config.training.val_frequency,
        "radius_threshold_l2": config.training.radius_threshold_l2,
        "radius_weight_l2": config.training.radius_weight_l2,
        "center_distance_threshold_l2": config.training.center_distance_threshold_l2,
        "center_distance_weight_l2": config.training.center_distance_weight_l2,
    }
    if values["val_dir"] == "":
        values["val_dir"] = None
    return argparse.Namespace(**values)

if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parse_args()
    raise SystemExit

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.profiler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import ClassPairFishDataset, FishDataset
from model import OpenWorldEncoder

# Enable CUDNN benchmark for potential speedup if input sizes are constant
torch.backends.cudnn.benchmark = True

class TimeTracker:
    """Track training progress and estimate remaining runtime."""

    def __init__(self, total_epochs: int, batches_per_epoch: int) -> None:
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.total_steps = total_epochs * batches_per_epoch

        self.training_start_time = time.time()
        self.epoch_start_times: list[float] = []
        self.step_start_time: float | None = None
        self.step_times: list[float] = []
        self.epoch_times: list[float] = []

        self.current_epoch = 0
        self.current_step_in_epoch = 0
        self.total_steps_completed = 0

    def start_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        self.current_step_in_epoch = 0
        self.epoch_start_times.append(time.time())

    def start_step(self) -> None:
        self.step_start_time = time.time()

    def complete_step(self) -> None:
        if self.step_start_time is not None:
            step_duration = time.time() - self.step_start_time
            self.step_times.append(step_duration)
            self.current_step_in_epoch += 1
            self.total_steps_completed += 1

    def complete_epoch(self) -> None:
        if self.epoch_start_times:
            epoch_duration = time.time() - self.epoch_start_times[-1]
            self.epoch_times.append(epoch_duration)

    def get_step_progress_info(self) -> dict[str, float | str] | None:
        if not self.step_times:
            return None

        recent_steps = min(50, len(self.step_times))
        avg_step_time = sum(self.step_times[-recent_steps:]) / recent_steps
        remaining_steps_in_epoch = self.batches_per_epoch - self.current_step_in_epoch
        remaining_time_in_epoch = remaining_steps_in_epoch * avg_step_time
        remaining_epochs = self.total_epochs - self.current_epoch
        remaining_steps_total = (
            remaining_epochs * self.batches_per_epoch + remaining_steps_in_epoch
        )
        remaining_time_total = remaining_steps_total * avg_step_time
        current_step_duration = (
            time.time() - self.step_start_time if self.step_start_time else 0
        )

        return {
            "step_progress": f"{self.current_step_in_epoch + 1}/{self.batches_per_epoch}",
            "epoch_progress": f"{self.current_epoch + 1}/{self.total_epochs}",
            "total_progress": f"{self.total_steps_completed + 1}/{self.total_steps}",
            "current_step_duration": current_step_duration,
            "avg_step_time": avg_step_time,
            "remaining_time_in_epoch": remaining_time_in_epoch,
            "remaining_time_total": remaining_time_total,
            "completion_percentage": (
                (self.total_steps_completed + 1) / self.total_steps
            )
            * 100,
        }

    def get_epoch_progress_info(self) -> dict[str, float] | None:
        if not self.epoch_times:
            return None

        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.total_epochs - (self.current_epoch + 1)
        estimated_remaining_time = remaining_epochs * avg_epoch_time
        total_elapsed = time.time() - self.training_start_time
        estimated_completion_time = (
            self.training_start_time + total_elapsed + estimated_remaining_time
        )
        current_epoch_duration = (
            time.time() - self.epoch_start_times[-1] if self.epoch_start_times else 0
        )

        return {
            "current_epoch_duration": current_epoch_duration,
            "avg_epoch_time": avg_epoch_time,
            "total_elapsed": total_elapsed,
            "estimated_remaining_time": estimated_remaining_time,
            "eta": estimated_completion_time,
            "completion_percentage": (
                (self.current_epoch + 1) / self.total_epochs
            )
            * 100,
        }

    @staticmethod
    def format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

    @staticmethod
    def format_eta(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")


def setup_logger(
    rank: int,
    save_path: str = "models/fish_classifier.pth",
    logger_name: str | None = None,
) -> logging.Logger:
    logger = logging.getLogger(logger_name or __name__)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers.clear()

    formatter = logging.Formatter(
        f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if rank == 0:
        log_dir = os.path.dirname(save_path)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_log_{timestamp}.log")

        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Training logs will be saved to: {log_file}")

    return logger


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == -1 or local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return True, rank, local_rank, world_size, device


def create_data_loaders(
    args: Any,
    is_distributed: bool,
    rank: int,
    world_size: int,
    logger_name: str | None = None,
) -> tuple[DataLoader, DataLoader | None, DistributedSampler | None]:
    logger = logging.getLogger(logger_name or __name__)

    if rank == 0:
        logger.info(f"Loading training data from {args.train_dir}")
        if args.samples_per_class:
            logger.info(f"Using at most {args.samples_per_class} samples per class")

    val_batch_size = args.batch_size * 2
    train_workers = args.num_workers
    val_workers = max(2, args.num_workers // 2)

    if args.use_class_pairs:
        if rank == 0:
            logger.info("Using class-paired BYOL approach with images from same class...")
        train_dataset, val_dataset = ClassPairFishDataset.get_datasets(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            image_size=args.image_size,
            samples_per_class=args.samples_per_class,
        )
    else:
        if rank == 0:
            logger.info("Using standard BYOL approach with augmentations...")
        train_dataset, val_dataset = FishDataset.get_datasets(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            image_size=args.image_size,
            samples_per_class=args.samples_per_class,
        )

    if rank == 0:
        logger.info(f"Training dataset size: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation dataset size: {len(val_dataset)}")
        logger.info(f"Training batch size per GPU: {args.batch_size}")
        logger.info(f"Validation batch size per GPU: {val_batch_size}")
        logger.info(f"Training workers per GPU: {train_workers}")
        logger.info(f"Validation workers per GPU: {val_workers}")
        logger.info(f"Drop last incomplete batch: {args.drop_last}")

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=args.drop_last,
        )
        val_sampler = (
            DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            if val_dataset
            else None
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        num_workers=train_workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True if train_workers > 0 else False,
        prefetch_factor=2 if train_workers > 0 else None,
        drop_last=args.drop_last,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=True,
            sampler=val_sampler,
            persistent_workers=True if val_workers > 0 else False,
            prefetch_factor=2 if val_workers > 0 else None,
            drop_last=False,
        )

    return train_loader, val_loader, train_sampler


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def _signal_handler(signum: int, frame: FrameType | None) -> None:
    print(f"Received signal {signum}, cleaning up...")
    cleanup()
    sys.exit(0)


def main():
    install_signal_handlers()
    
    args = parse_args()

    # Setup distributed training
    is_distributed, rank, local_rank, world_size, device = setup_distributed()
    
    # Setup logger
    logger = setup_logger(rank, args.save_path, logger_name=__name__)
    
    if rank == 0:
        logger.info(f"Using device: {device}")
        logger.info(f"Distributed training: {is_distributed}")
        logger.info(f"World size: {world_size}")
        if is_distributed:
            logger.info(f"Rank: {rank}, Local rank: {local_rank}")

    try:
        # Create data loaders
        train_loader, val_loader, train_sampler = create_data_loaders(
            args, is_distributed, rank, world_size, logger_name=__name__
        )
        
        if rank == 0:
            if val_loader is not None:
                logger.info(f"Validation will run every {args.val_frequency} epochs")
            else:
                logger.info("No validation data provided, training without validation")

        # Create model with L2 regularization parameters
        model = OpenWorldEncoder(
            model_path=args.model_path,
            backbone=args.backbone,
            radius_threshold_l2=args.radius_threshold_l2,
            radius_weight_l2=args.radius_weight_l2,
            center_distance_threshold_l2=args.center_distance_threshold_l2,
            center_distance_weight_l2=args.center_distance_weight_l2
        ).to(device)
        
        if rank == 0:
            logger.info(f"Using {args.backbone} as backbone network")
            logger.info(f"Regularization Parameters:")
            logger.info(f"L2 Distance Parameters:")
            logger.info(f"  - Radius threshold (L2): {args.radius_threshold_l2}")
            logger.info(f"  - Radius weight (L2): {args.radius_weight_l2}")
            logger.info(f"  - Center distance threshold (L2): {args.center_distance_threshold_l2}")
            logger.info(f"  - Center distance weight (L2): {args.center_distance_weight_l2}")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")

        # Wrap model with DDP if distributed
        if is_distributed:
            model = DDP(
                model, 
                device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=False  # Set to True if you have unused parameters
            )
            if rank == 0:
                logger.info("Model wrapped with DistributedDataParallel")
        elif torch.cuda.device_count() > 1:
            logger.info(f"Found {torch.cuda.device_count()} GPUs. Using DataParallel")
            model = nn.DataParallel(model)
        
        # Train model
        if rank == 0:
            logger.info(f"Starting training for {args.epochs} epochs...")
        
        save_path = _train_byol(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_path=args.save_path,
            use_class_pairs=args.use_class_pairs,
            device=device,
            is_distributed=is_distributed,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            profile_batches=args.profile_batches,
            gradient_clip=args.gradient_clip,
            warmup_epochs=args.warmup_epochs,
            args=args # Pass args to _train_byol
        )
        
        if rank == 0:
            logger.info(f"Training completed. Model saved to {args.save_path}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise e
    finally:
        cleanup()

def _train_byol(model, train_loader, val_loader, train_sampler, epochs, lr, weight_decay, 
                save_path, use_class_pairs, device, is_distributed, rank, local_rank, 
                world_size, profile_batches=0, gradient_clip=1.0, warmup_epochs=5, args=None):
    """
    Enhanced training function using BYOL with comprehensive multi-GPU support and detailed time tracking
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        train_sampler: DistributedSampler for training data
        epochs: Number of training epochs
        lr: Base learning rate
        weight_decay: Weight decay for optimizer
        save_path: Path to save model (directory path)
        use_class_pairs: Whether to use class-paired BYOL
        device: The device to use for training
        is_distributed: Boolean indicating if distributed training is used
        rank: Global rank of the current process
        local_rank: Local rank of the current process
        world_size: Total number of processes
        profile_batches: Number of batches to profile per epoch
        gradient_clip: Gradient clipping value
        warmup_epochs: Number of warmup epochs
    """
    logger = logging.getLogger(__name__)
    
    # Initialize enhanced time tracker
    time_tracker = TimeTracker(epochs, len(train_loader))
    
    if rank == 0:
        start_time_str = datetime.fromtimestamp(time_tracker.training_start_time).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Training started at: {start_time_str}")
        logger.info(f"Total epochs: {epochs}, Batches per epoch: {len(train_loader)}, Total steps: {time_tracker.total_steps}")
        if val_loader is not None:
            logger.info(f"Validation will run every {args.val_frequency} epochs")

    # Get the actual model for optimization (unwrap DDP/DataParallel)
    model_to_optimize = model.module if isinstance(model, (DDP, nn.DataParallel)) else model

    # Scale learning rate by world size for distributed training
    scaled_lr = lr * world_size if is_distributed else lr
    
    # Setup optimizer with different learning rates for different components
    optimizer = torch.optim.AdamW([
        {'params': model_to_optimize.online_backbone.parameters(), 'lr': scaled_lr * 0.1},
        {'params': model_to_optimize.online_projector.parameters(), 'lr': scaled_lr},
        {'params': model_to_optimize.online_predictor.parameters(), 'lr': scaled_lr}
    ], weight_decay=weight_decay)
    
    # Learning rate scheduler with warmup
    def get_lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
    
    # Setup AMP for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 15  # Increased patience for multi-GPU training
    patience_counter = 0
    best_epoch = 0
    
    # Training metrics tracking
    train_losses = []
    val_losses = []
    
    if rank == 0:
        logger.info(f"Optimizer setup: Base LR={lr}, Scaled LR={scaled_lr}")
        logger.info(f"Warmup epochs: {warmup_epochs}, Total epochs: {epochs}")
        logger.info(f"Gradient clipping: {gradient_clip}")
        logger.info(f"Using mixed precision training: {scaler.is_enabled()}")
    
    # Function to save model checkpoint
    def save_checkpoint(epoch, val_loss):
        if rank != 0:
            return
            
        # Create checkpoint directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Prepare model state
        model_state = model.module.state_dict() if isinstance(model, (DDP, nn.DataParallel)) else model.state_dict()
        
        # Calculate current training duration
        current_training_duration = time.time() - time_tracker.training_start_time
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses if val_loader is not None else None,
            'best_val_loss': val_loss,
            'training_start_time': time_tracker.training_start_time,
            'training_duration_seconds': current_training_duration,
            'epoch_times': time_tracker.epoch_times,
            'step_times': time_tracker.step_times,
            'args': {
                'backbone': model_to_optimize.backbone_name,
                'lr': lr,
                'world_size': world_size
            }
        }
        
        # Save current epoch model
        current_path = os.path.join(save_path, f'fish_classifier_epoch_{epoch+1}.pth')
        torch.save(checkpoint, current_path)
        logger.info(f"💾 Saved model at epoch {epoch+1} to {current_path}")
        
        # If this is the best model so far, save it as best model
        if val_loss <= getattr(save_checkpoint, 'best_val_loss', float('inf')):
            save_checkpoint.best_val_loss = val_loss
            best_path = os.path.join(save_path, f'fish_classifier_best_epoch_{epoch+1}_loss_{val_loss:.4f}.pth')
            # Create a copy of the checkpoint as best model
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 New best model! Saved at epoch {epoch+1} with validation loss: {val_loss:.4f} to {best_path}")
            
            # Remove previous best model if it exists
            if hasattr(save_checkpoint, 'prev_best_path') and os.path.exists(save_checkpoint.prev_best_path):
                os.remove(save_checkpoint.prev_best_path)
            save_checkpoint.prev_best_path = best_path
        

    # Training loop
    for epoch in range(epochs):
        # Start epoch tracking
        time_tracker.start_epoch(epoch)
        
        # Set epoch for distributed sampler to ensure different shuffling
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Training phase
        model.train()
        total_loss_sum = 0
        byol_loss_sum = 0
        radius_loss_l2_sum = 0
        center_distance_loss_l2_sum = 0
        batch_count = 0
        
        # Setup profiler if needed
        prof = None
        if rank == 0 and profile_batches > 0 and epoch == 0:
            logger.info(f"Profiling first {profile_batches} batches of epoch 0...")
            prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=1, warmup=1, active=profile_batches, repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_traces/train_byol'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            prof.start()

        # Training loop
        for batch_idx, batch_data in enumerate(train_loader):
            # Start step tracking
            time_tracker.start_step()
            
            # Stop profiling after specified batches
            if rank == 0 and prof and batch_idx >= (1 + 1 + profile_batches):
                prof.stop()
                logger.info(f"Profiling finished. Trace saved to ./profiler_traces/train_byol")
                prof = None
            
            # Prepare batch data
            if use_class_pairs:
                anchor_images, positive_images, labels = batch_data
                anchor_images = anchor_images.to(device, non_blocking=True)
                positive_images = positive_images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            else:
                src_images, labels = batch_data
                src_images = src_images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                if use_class_pairs:
                    total_loss, byol_loss, radius_loss_l2, center_distance_loss_l2 = model(anchor_images, positive_images, labels)
                else:
                    total_loss, byol_loss, radius_loss_l2, center_distance_loss_l2 = model(src_images, labels=labels)
                
                # Average loss if using DataParallel (not DDP)
                if not is_distributed and isinstance(model, nn.DataParallel):
                    total_loss = total_loss.mean()
                    byol_loss = byol_loss.mean()
                    radius_loss_l2 = radius_loss_l2.mean()
                    center_distance_loss_l2 = center_distance_loss_l2.mean()

            # Backward pass with gradient scaling
            scaler.scale(total_loss).backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Update target network (EMA update)
            if isinstance(model, (DDP, nn.DataParallel)):
                model.module.update_target_network()
            else:
                model.update_target_network()

            # Track all losses
            total_loss_val = total_loss.item()
            byol_loss_val = byol_loss.item()
            radius_loss_l2_val = radius_loss_l2.item()
            center_distance_loss_l2_val = center_distance_loss_l2.item()
            
            total_loss_sum += total_loss_val
            byol_loss_sum += byol_loss_val
            radius_loss_l2_sum += radius_loss_l2_val
            center_distance_loss_l2_sum += center_distance_loss_l2_val
            batch_count += 1
            
            # Complete step tracking
            time_tracker.complete_step()
            
            # Enhanced progress logging with time estimates
            if rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                
                # Get step progress info
                step_info = time_tracker.get_step_progress_info()
                
                # Display progress every 10 steps or at important milestones
                should_log = (
                    batch_idx % 10 == 0 or  # Every 10 steps
                    batch_idx == len(train_loader) - 1 or  # Last batch
                    batch_idx < 5  # First few batches
                )
                
                if should_log and step_info:
                    # Format time information
                    current_step_time = TimeTracker.format_time(step_info['current_step_duration'])
                    avg_step_time = TimeTracker.format_time(step_info['avg_step_time'])
                    remaining_epoch_time = TimeTracker.format_time(step_info['remaining_time_in_epoch'])
                    remaining_total_time = TimeTracker.format_time(step_info['remaining_time_total'])
                    
                    # Basic progress info with detailed losses
                    logger.info(f"Epoch {step_info['epoch_progress']} | "
                               f"Step {step_info['step_progress']} | "
                               f"Total Loss: {total_loss_val:.4f} | "
                               f"BYOL Loss: {byol_loss_val:.4f} | "
                               f"Radius L2: {radius_loss_l2_val:.4f} | "
                               f"Center L2: {center_distance_loss_l2_val:.4f} | "
                               f"LR: {current_lr:.2e}")
                    
                    # Time information
                    logger.info(f"Step: {current_step_time} | "
                               f"Avg: {avg_step_time} | "
                               f"Epoch ETA: {remaining_epoch_time} | "
                               f"Total ETA: {remaining_total_time} | "
                               f"Progress: {step_info['completion_percentage']:.1f}%")
                    
                # Quick progress for every step (less verbose)
                elif batch_idx % 50 == 0 and step_info:
                    remaining_total_time = TimeTracker.format_time(step_info['remaining_time_total'])
                    logger.info(f"E{epoch+1}/{epochs} B{batch_idx+1}/{len(train_loader)} | "
                               f"Total: {total_loss_val:.4f} | "
                               f"BYOL: {byol_loss_val:.4f} | "
                               f"Radius L2: {radius_loss_l2_val:.4f} | "
                               f"Center L2: {center_distance_loss_l2_val:.4f} | "
                               f"ETA: {remaining_total_time} | "
                               f"{step_info['completion_percentage']:.1f}%")

            # Step profiler if active
            if rank == 0 and prof:
                prof.step()
        
        # Complete epoch tracking
        time_tracker.complete_epoch()
        
        # Cleanup profiler
        if rank == 0 and prof:
            prof.stop()
            logger.info(f"Profiling finished at end of epoch. Trace saved to ./profiler_traces/train_byol")
            prof = None

        # Calculate average training losses
        avg_total_loss = total_loss_sum / batch_count
        avg_byol_loss = byol_loss_sum / batch_count
        avg_radius_loss_l2 = radius_loss_l2_sum / batch_count
        avg_center_distance_loss_l2 = center_distance_loss_l2_sum / batch_count
        
        # Synchronize training losses across all processes
        if is_distributed:
            avg_total_loss_tensor = torch.tensor(avg_total_loss, device=device)
            avg_byol_loss_tensor = torch.tensor(avg_byol_loss, device=device)
            avg_radius_loss_l2_tensor = torch.tensor(avg_radius_loss_l2, device=device)
            avg_center_distance_loss_l2_tensor = torch.tensor(avg_center_distance_loss_l2, device=device)
            
            dist.all_reduce(avg_total_loss_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(avg_byol_loss_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(avg_radius_loss_l2_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(avg_center_distance_loss_l2_tensor, op=dist.ReduceOp.AVG)
            
            avg_total_loss = avg_total_loss_tensor.item()
            avg_byol_loss = avg_byol_loss_tensor.item()
            avg_radius_loss_l2 = avg_radius_loss_l2_tensor.item()
            avg_center_distance_loss_l2 = avg_center_distance_loss_l2_tensor.item()

        train_losses.append(avg_total_loss)
        
        # Enhanced epoch summary with time estimates
        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            epoch_info = time_tracker.get_epoch_progress_info()
            
            if epoch_info:
                # Format time information
                epoch_duration = TimeTracker.format_time(epoch_info['current_epoch_duration'])
                avg_epoch_time = TimeTracker.format_time(epoch_info['avg_epoch_time'])
                total_elapsed = TimeTracker.format_time(epoch_info['total_elapsed'])
                remaining_time = TimeTracker.format_time(epoch_info['estimated_remaining_time'])
                eta = TimeTracker.format_eta(epoch_info['eta'])
                
                logger.info("="*80)
                logger.info(f"EPOCH {epoch+1}/{epochs} COMPLETED")
                logger.info(f"📈 Total Loss: {avg_total_loss:.4f} | "
                           f"BYOL Loss: {avg_byol_loss:.4f} | "
                           f"Radius L2: {avg_radius_loss_l2:.4f} | "
                           f"Center L2: {avg_center_distance_loss_l2:.4f} | "
                           f"LR: {current_lr:.2e}")
                logger.info(f"This Epoch: {epoch_duration} | Avg/Epoch: {avg_epoch_time}")
                logger.info(f"🕐 Elapsed: {total_elapsed} | Remaining: {remaining_time} | ETA: {eta}")
                logger.info(f"Progress: {epoch_info['completion_percentage']:.1f}% complete")
                logger.info("="*80)

        # Validation phase
        if val_loader is not None and (epoch + 1) % args.val_frequency == 0:
            # Track validation time
            val_start_time = time.time()
            if rank == 0:
                logger.info(f"🔍 Starting validation for epoch {epoch+1}...")
                logger.info(f"Validation batches: {len(val_loader)}")
                
                model.eval()
                val_total_loss = 0
                val_byol_loss = 0
                val_radius_loss_l2 = 0
                val_center_distance_loss_l2 = 0
                val_batch_count = 0
                
                # Process validation in chunks to show progress
                chunk_size = max(1, len(val_loader) // 10)  # Show progress 10 times
                chunk_losses = []
                
                with torch.no_grad():
                    for chunk_start in range(0, len(val_loader), chunk_size):
                        chunk_end = min(chunk_start + chunk_size, len(val_loader))
                        chunk_total_loss = 0
                        chunk_byol_loss = 0
                        chunk_radius_loss_l2 = 0
                        chunk_center_distance_loss_l2 = 0
                        chunk_count = 0
                        
                        # Process one chunk of batches
                        for val_batch_idx in range(chunk_start, chunk_end):
                            batch_data = next(iter(val_loader))
                            
                            if use_class_pairs:
                                anchor_images, positive_images, labels = batch_data
                                anchor_images = anchor_images.to(device, non_blocking=True)
                                positive_images = positive_images.to(device, non_blocking=True)
                                labels = labels.to(device, non_blocking=True)
                                
                                with torch.amp.autocast('cuda'):
                                    total_loss, byol_loss, radius_loss_l2, center_distance_loss_l2 = model(anchor_images, positive_images, labels)
                            else:
                                src_images, labels = batch_data
                                src_images = src_images.to(device, non_blocking=True)
                                labels = labels.to(device, non_blocking=True)
                                
                                with torch.amp.autocast('cuda'):
                                    total_loss, byol_loss, radius_loss_l2, center_distance_loss_l2 = model(src_images, labels=labels)
                            
                            # Average loss if using DataParallel
                            if not is_distributed and isinstance(model, nn.DataParallel):
                                total_loss = total_loss.mean()
                                byol_loss = byol_loss.mean()
                                radius_loss_l2 = radius_loss_l2.mean()
                                center_distance_loss_l2 = center_distance_loss_l2.mean()

                            chunk_total_loss += total_loss.item()
                            chunk_byol_loss += byol_loss.item()
                            chunk_radius_loss_l2 += radius_loss_l2.item()
                            chunk_center_distance_loss_l2 += center_distance_loss_l2.item()
                            chunk_count += 1
                        
                        # Update overall validation metrics
                        val_total_loss += chunk_total_loss
                        val_byol_loss += chunk_byol_loss
                        val_radius_loss_l2 += chunk_radius_loss_l2
                        val_center_distance_loss_l2 += chunk_center_distance_loss_l2
                        val_batch_count += chunk_count
                        
                        # Show progress after each chunk
                        if rank == 0:
                            progress = (chunk_end) / len(val_loader) * 100
                            avg_chunk_total = chunk_total_loss / chunk_count
                            avg_chunk_byol = chunk_byol_loss / chunk_count
                            avg_chunk_radius_l2 = chunk_radius_loss_l2 / chunk_count
                            avg_chunk_center_distance_l2 = chunk_center_distance_loss_l2 / chunk_count
                            
                            val_elapsed = time.time() - val_start_time
                            val_eta = (val_elapsed / (chunk_end)) * (len(val_loader) - chunk_end)
                            
                            logger.info(
                                f"Validation progress: {chunk_end}/{len(val_loader)} "
                                f"({progress:.1f}%) | "
                                f"Chunk Loss: {avg_chunk_total:.4f} | "
                                f"BYOL: {avg_chunk_byol:.4f} | "
                                f"Radius L2: {avg_chunk_radius_l2:.4f} | "
                                f"Center L2: {avg_chunk_center_distance_l2:.4f} | "
                                f"ETA: {TimeTracker.format_time(val_eta)}"
                            )
                
                # Calculate validation duration
                val_duration = time.time() - val_start_time
                
                # Calculate average validation losses
                avg_val_total_loss = val_total_loss / val_batch_count
                avg_val_byol_loss = val_byol_loss / val_batch_count
                avg_val_radius_loss_l2 = val_radius_loss_l2 / val_batch_count
                avg_val_center_distance_loss_l2 = val_center_distance_loss_l2 / val_batch_count
                
                # Synchronize validation losses across all processes
                if is_distributed:
                    avg_val_total_loss_tensor = torch.tensor(avg_val_total_loss, device=device)
                    avg_val_byol_loss_tensor = torch.tensor(avg_val_byol_loss, device=device)
                    avg_val_radius_loss_l2_tensor = torch.tensor(avg_val_radius_loss_l2, device=device)
                    avg_val_center_distance_loss_l2_tensor = torch.tensor(avg_val_center_distance_loss_l2, device=device)
                    
                    dist.all_reduce(avg_val_total_loss_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(avg_val_byol_loss_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(avg_val_radius_loss_l2_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(avg_val_center_distance_loss_l2_tensor, op=dist.ReduceOp.AVG)
                    
                    avg_val_total_loss = avg_val_total_loss_tensor.item()
                    avg_val_byol_loss = avg_val_byol_loss_tensor.item()
                    avg_val_radius_loss_l2 = avg_val_radius_loss_l2_tensor.item()
                    avg_val_center_distance_loss_l2 = avg_val_center_distance_loss_l2_tensor.item()

                val_losses.append(avg_val_total_loss)

                # Early stopping and model saving (only on main process)
                if rank == 0:
                    logger.info(f"✅ Validation completed in {TimeTracker.format_time(val_duration)}")
                    logger.info(f"Validation Total Loss: {avg_val_total_loss:.4f} | "
                               f"BYOL Loss: {avg_val_byol_loss:.4f} | "
                               f"Radius L2: {avg_val_radius_loss_l2:.4f} | "
                               f"Center L2: {avg_val_center_distance_loss_l2:.4f}")
                    
                    if avg_val_total_loss < best_val_loss:
                        best_val_loss = avg_val_total_loss
                        best_epoch = epoch
                        patience_counter = 0
                        
                        # Save best model
                        save_checkpoint(epoch, best_val_loss)
                    else:
                        patience_counter += 1
                        logger.info(f"⚠️  Validation loss did not improve. Patience: {patience_counter}/{patience}")
                        save_checkpoint(epoch, avg_val_total_loss)
                        if patience_counter >= patience:
                            logger.info(f"🛑 Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}")
                            break
        else:
            if rank == 0 and val_loader is not None:
                logger.info(f"Skipping validation for epoch {epoch+1} (next validation at epoch {((epoch + 1) // args.val_frequency + 1) * args.val_frequency})")
            
            # Save checkpoint based on save_frequency
            if rank == 0:
                if (epoch + 1) % args.save_frequency == 0:
                    logger.info(f"💾 Saving model at epoch {epoch+1} (save frequency: {args.save_frequency})")
                    save_checkpoint(epoch, float('inf'))  # Use infinity as we don't have validation loss
        
        # Step the scheduler
        scheduler.step()
        
        # Synchronize all processes before next epoch
        if is_distributed:
            dist.barrier()
    
    # Final model loading and saving
    if rank == 0:
        # Calculate final training duration
        final_training_duration = time.time() - time_tracker.training_start_time
        final_duration_str = str(timedelta(seconds=int(final_training_duration)))
        
        # Enhanced final training summary
        logger.info("="*100)
        logger.info("🏁 TRAINING COMPLETED!")
        logger.info("="*100)
        logger.info(f"Total Training Time: {final_duration_str}")
        
        if time_tracker.epoch_times:
            avg_epoch_time_final = sum(time_tracker.epoch_times) / len(time_tracker.epoch_times)
            avg_epoch_str = str(timedelta(seconds=int(avg_epoch_time_final)))
            logger.info(f"Average Time per Epoch: {avg_epoch_str}")
            
            # Additional statistics
            fastest_epoch = min(time_tracker.epoch_times)
            slowest_epoch = max(time_tracker.epoch_times)
            logger.info(f"⚡ Fastest Epoch: {TimeTracker.format_time(fastest_epoch)}")
            logger.info(f"🐌 Slowest Epoch: {TimeTracker.format_time(slowest_epoch)}")
        
        if time_tracker.step_times:
            avg_step_time = sum(time_tracker.step_times) / len(time_tracker.step_times)
            logger.info(f"🔄 Average Time per Step: {TimeTracker.format_time(avg_step_time)}")
            logger.info(f"📈 Total Steps Completed: {len(time_tracker.step_times)}")
            
            # Steps per second
            steps_per_second = len(time_tracker.step_times) / final_training_duration
            logger.info(f"🚀 Training Speed: {steps_per_second:.2f} steps/second")
        
        logger.info(f"📅 Training Started: {datetime.fromtimestamp(time_tracker.training_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"🏁 Training Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if val_loader is not None:
            logger.info(f"✅ Best model saved at epoch {best_epoch+1}")
            logger.info(f"🏆 Best validation loss: {best_val_loss:.4f}")
            logger.info(f"📚 Training history: {len(train_losses)} epochs")
        else:
            logger.info(f"✅ Training completed without validation")
            logger.info(f"📚 Training history: {len(train_losses)} epochs")
        
        logger.info("="*100)

    # Final synchronization
    if is_distributed:
        dist.barrier()

    # Return the path to the best model if validation was used, otherwise the latest model
    if val_loader is not None and rank == 0:
        return os.path.join(save_path, f'fish_classifier_best_epoch_{best_epoch+1}_loss_{best_val_loss:.4f}.pth')
    else:
        return os.path.join(save_path, f'fish_classifier_epoch_{epochs}.pth')

if __name__ == "__main__":
    main()
