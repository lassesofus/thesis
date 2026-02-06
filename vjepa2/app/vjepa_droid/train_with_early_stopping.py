"""
Modified V-JEPA training with early stopping support.

This is a modified version of app/vjepa_droid/train.py that adds:
- Validation data loader
- Validation loss evaluation after each epoch
- Early stopping based on validation loss
- Best model checkpointing
"""

import copy
import gc
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*UnsupportedFieldAttributeWarning.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*timm.models.layers.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.cuda.amp.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.backends.cuda.sdp_kernel.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*requires_grad.*scalar.*')

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.vjepa_droid.droid import init_data
from app.vjepa_droid.transforms import make_transforms
from app.vjepa_droid.utils import init_opt, init_video_model, load_checkpoint, load_pretrained
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

# Constants
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
_GLOBAL_SEED = 0

random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)

# Override the main function to add early stopping
def main_with_early_stopping(args, resume_preempt=False):
    """Modified main function with early stopping."""

    # Get early stopping parameters from config (with defaults)
    cfgs_meta = args.get("meta", {})
    early_stopping_patience = cfgs_meta.get("early_stopping_patience", 10)
    val_freq = cfgs_meta.get("val_freq", 1)  # Validate every N epochs
    min_delta = cfgs_meta.get("early_stopping_min_delta", 0.0001)

    # Check if validation data is configured
    cfgs_val_data = args.get("val_data", None)
    if cfgs_val_data is None:
        logger.warning("No val_data in config! Running without early stopping.")
        # Fall back to original training
        from app.vjepa_droid.train import main as original_main
        return original_main(args, resume_preempt)

    logger.info(f"Early stopping enabled: patience={early_stopping_patience}, min_delta={min_delta}, val_freq={val_freq}")

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    r_file = cfgs_meta.get("resume_checkpoint", None)
    p_file = cfgs_meta.get("pretrain_checkpoint", None)
    load_predictor = cfgs_meta.get("load_predictor", False)
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    target_encoder_key = cfgs_meta.get("target_encoder_key", "target_encoder")
    load_encoder = cfgs_meta.get("load_encoder", True)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
    uniform_power = cfgs_model.get("uniform_power", False)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    use_extrinsics = cfgs_model.get("use_extrinsics", False)

    # -- DATA
    cfgs_data = args.get("data")
    datasets = cfgs_data.get("datasets", [])
    dataset_path = datasets[0]
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    camera_frame = cfgs_data.get("camera_frame", False)
    camera_views = cfgs_data.get("camera_views", ["left_mp4_path"])
    stereo_view = cfgs_data.get("stereo_view", False)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    # -- VAL DATA
    val_datasets = cfgs_val_data.get("datasets", [])
    val_dataset_path = val_datasets[0]
    val_batch_size = cfgs_val_data.get("batch_size", batch_size)
    val_num_workers = cfgs_val_data.get("num_workers", num_workers)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    horizontal_flip = cfgs_data_aug.get("horizontal_flip", False)
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    normalize_reps = cfgs_loss.get("normalize_reps")
    auto_steps = min(cfgs_loss.get("auto_steps", 1), max_num_frames)
    # --
    tokens_per_frame = int((crop_size // patch_size) ** 2)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    anneal = cfgs_opt.get("anneal")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    enc_lr_scale = cfgs_opt.get("enc_lr_scale", 1.0) # Defaults to 1 as this is not set in any of the Droid training config files. 
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    # Create output folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # -- Initialize wandb (only on main process)
    run_name = Path(folder).name  # e.g., "4.8.vitg16-256px-8f_025pct"
    if rank == 0:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "vjepa2-training"),
            entity=os.environ.get("WANDB_ENTITY", None),
            name=run_name,
            config=args,
            resume="allow",
        )

    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    best_path = os.path.join(folder, "best.pt")  # NEW: Best model path
    val_log_file = os.path.join(folder, f"val_log_r{rank}.csv")  # NEW: Validation log
    resume_path = os.path.join(folder, r_file) if r_file is not None else latest_path
    if not os.path.exists(resume_path):
        resume_path = None

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
        mode="+a",
    )

    # -- NEW: make validation csv_logger
    val_csv_logger = CSVLogger(
        val_log_file,
        ("%d", "epoch"),
        ("%.5f", "val_loss"),
        ("%.5f", "val_jloss"),
        ("%.5f", "val_sloss"),
        mode="+a",
    )

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        device=device,
        patch_size=patch_size,
        max_num_frames=512,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        action_embed_dim=7,
        pred_is_frame_causal=pred_is_frame_causal,
        use_extrinsics=use_extrinsics,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    target_encoder = copy.deepcopy(encoder)

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()

    video_collator = torch.utils.data.default_collate
    transform = make_transforms(
        random_horizontal_flip=horizontal_flip,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- init TRAINING data-loader
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data_path=dataset_path,
        batch_size=batch_size,
        frames_per_clip=max_num_frames,
        tubelet_size=1,
        fps=fps,
        camera_views=camera_views,
        camera_frame=camera_frame,
        stereo_view=stereo_view,
        transform=transform,
        collator=video_collator,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        rank=rank,
    )

    # -- NEW: init VALIDATION data-loader (no augmentation)
    val_transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=[1.0, 1.0],
        random_resize_scale=[1.777, 1.777],
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )

    (val_loader, val_sampler) = init_data(
        data_path=val_dataset_path,
        batch_size=val_batch_size,
        frames_per_clip=max_num_frames,
        tubelet_size=1,
        fps=fps,
        camera_views=camera_views,
        camera_frame=camera_frame,
        stereo_view=stereo_view,
        transform=val_transform,
        collator=video_collator,
        num_workers=val_num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        rank=rank,
    )
    logger.info(f"Validation loader created with {len(val_loader)} batches")

    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        enc_lr_scale=enc_lr_scale,
        iterations_per_epoch=ipe,
        anneal=anneal,
        warmup=warmup,
        num_epochs=num_epochs,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )
    # Wrap in DDP only if distributed training is actually initialized
    # For single GPU, distributed is not initialized by init_distributed()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)
        target_encoder = DistributedDataParallel(target_encoder)
        logger.info("Using DistributedDataParallel")
    else:
        logger.info("Single GPU mode - not using DistributedDataParallel")

    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- load pretrained weights
    # For single GPU (non-DDP), we need to handle 'module.' prefix in checkpoints
    using_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()

    if p_file and load_encoder and not using_ddp:
        # Custom loading for single GPU to strip 'module.' prefix
        logger.info(f"Loading pretrained weights (single GPU mode) from {p_file}")
        checkpoint = torch.load(p_file, map_location=device)

        # Load encoder - strip both 'module.' and 'backbone.' prefixes
        pretrained_dict = checkpoint[context_encoder_key]
        pretrained_dict = {k.replace("module.", "").replace("backbone.", ""): v
                          for k, v in pretrained_dict.items()}
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"Loaded pretrained encoder with msg: {msg}")

        # Load target encoder
        pretrained_dict = checkpoint[target_encoder_key]
        pretrained_dict = {k.replace("module.", "").replace("backbone.", ""): v
                          for k, v in pretrained_dict.items()}
        msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"Loaded pretrained target encoder with msg: {msg}")

        # Load predictor if requested
        if load_predictor and "predictor" in checkpoint:
            pretrained_dict = checkpoint["predictor"]
            pretrained_dict = {k.replace("module.", "").replace("backbone.", ""): v
                              for k, v in pretrained_dict.items()}
            msg = predictor.load_state_dict(pretrained_dict, strict=False)
            logger.info(f"Loaded pretrained predictor with msg: {msg}")

        del checkpoint
    else:
        # Use original loading function for DDP or when not loading encoder
        encoder, predictor, target_encoder = load_pretrained(
            r_path=p_file,
            encoder=encoder,
            predictor=predictor,
            context_encoder_key=context_encoder_key,
            target_encoder_key=target_encoder_key,
            target_encoder=target_encoder,
            load_predictor=load_predictor,
            load_encoder=load_encoder,
        )

    start_epoch = 0
    # -- NEW: Early stopping state
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # -- load training checkpoint
    if os.path.exists(latest_path):
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=resume_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

        # Try to load early stopping state
        if os.path.exists(best_path):
            best_checkpoint = torch.load(best_path, map_location=device)
            best_val_loss = best_checkpoint.get('val_loss', float('inf'))
            epochs_without_improvement = best_checkpoint.get('epochs_without_improvement', 0)
            logger.info(f"Resumed early stopping state: best_val_loss={best_val_loss:.4f}, patience_counter={epochs_without_improvement}")

    def save_checkpoint(epoch, path, val_loss=None, is_best=False):
        if rank != 0:
            return

        # Get state dicts - .state_dict() on DDP models includes 'module.' prefix
        # For non-DDP models, we get the raw state dict
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        if val_loss is not None:
            save_dict["val_loss"] = val_loss
            save_dict["epochs_without_improvement"] = epochs_without_improvement
            save_dict["is_best"] = is_best
        try:
            torch.save(save_dict, path)
            if is_best:
                logger.info(f"★ Saved best model with val_loss={val_loss:.4f}")
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    # -- NEW: Validation function
    def validate():
        """Run validation and return average loss."""
        encoder.eval()
        predictor.eval()
        target_encoder.eval()

        val_loss_meter = AverageMeter()
        val_jloss_meter = AverageMeter()
        val_sloss_meter = AverageMeter()

        with torch.no_grad():
            for batch_idx, sample in enumerate(val_loader):
                try:
                    clips = sample[0].to(device, non_blocking=True)
                    actions = sample[1].to(device, dtype=torch.float, non_blocking=True)
                    states = sample[2].to(device, dtype=torch.float, non_blocking=True)
                    extrinsics = sample[3].to(device, dtype=torch.float, non_blocking=True)

                    with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                        # Target encoding
                        c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
                        h = target_encoder(c)
                        h = h.view(clips.size(0), max_num_frames, -1, h.size(-1)).flatten(1, 2)
                        if normalize_reps:
                            h = F.layer_norm(h, (h.size(-1),))

                        # Teacher-forced predictions
                        _z = h[:, :-tokens_per_frame]
                        z_tf = predictor(_z, actions, states[:, :-1], extrinsics[:, :-1])
                        if normalize_reps:
                            z_tf = F.layer_norm(z_tf, (z_tf.size(-1),))

                        # Auto-regressive predictions
                        _z = torch.cat([h[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]], dim=1)
                        for n in range(1, auto_steps):
                            _a = actions[:, :n + 1]
                            _s = states[:, :n + 1]
                            _e = extrinsics[:, :n + 1]
                            _z_nxt = predictor(_z, _a, _s, _e)[:, -tokens_per_frame:]
                            if normalize_reps:
                                _z_nxt = F.layer_norm(_z_nxt, (_z_nxt.size(-1),))
                            _z = torch.cat([_z, _z_nxt], dim=1)
                        z_ar = _z[:, tokens_per_frame:]

                        # Compute losses
                        _h_tf = h[:, tokens_per_frame : z_tf.size(1) + tokens_per_frame]
                        jloss = torch.mean(torch.abs(z_tf - _h_tf) ** loss_exp) / loss_exp

                        _h_ar = h[:, tokens_per_frame : z_ar.size(1) + tokens_per_frame]
                        sloss = torch.mean(torch.abs(z_ar - _h_ar) ** loss_exp) / loss_exp

                        loss = jloss + sloss

                    val_loss_meter.update(float(loss))
                    val_jloss_meter.update(float(jloss))
                    val_sloss_meter.update(float(sloss))

                except Exception as e:
                    logger.warning(f"Validation batch {batch_idx} failed: {e}")
                    continue

        encoder.train()
        predictor.train()
        target_encoder.train()

        return val_loss_meter.avg, val_jloss_meter.avg, val_sloss_meter.avg

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    # -- TRAINING LOOP (with early stopping)
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        jloss_meter = AverageMeter()
        sloss_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                        raise e

            def load_clips():
                clips = sample[0].to(device, non_blocking=True)
                actions = sample[1].to(device, dtype=torch.float, non_blocking=True)
                states = sample[2].to(device, dtype=torch.float, non_blocking=True)
                extrinsics = sample[3].to(device, dtype=torch.float, non_blocking=True)
                return (clips, actions, states, extrinsics)

            clips, actions, states, extrinsics = load_clips()
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target(c):
                    with torch.no_grad():
                        c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
                        h = target_encoder(c)
                        h = h.view(batch_size, max_num_frames, -1, h.size(-1)).flatten(1, 2)
                        if normalize_reps:
                            h = F.layer_norm(h, (h.size(-1),))
                        return h

                def forward_predictions(z):
                    def _step_predictor(_z, _a, _s, _e):
                        _z = predictor(_z, _a, _s, _e)
                        if normalize_reps:
                            _z = F.layer_norm(_z, (_z.size(-1),))
                        return _z

                    _z, _a, _s, _e = z[:, :-tokens_per_frame], actions, states[:, :-1], extrinsics[:, :-1]
                    z_tf = _step_predictor(_z, _a, _s, _e)

                    _z = torch.cat([z[:, : tokens_per_frame], z_tf[:, : tokens_per_frame]], dim=1)
                    for n in range(1, auto_steps):
                        _a, _s, _e = actions[:, : n + 1], states[:, : n + 1], extrinsics[:, : n + 1]
                        _z_nxt = _step_predictor(_z, _a, _s, _e)[:, -tokens_per_frame:]
                        _z = torch.cat([_z, _z_nxt], dim=1)
                    z_ar = _z[:, tokens_per_frame:]

                    return z_tf, z_ar

                def loss_fn(z, h):
                    _h = h[:, tokens_per_frame : z.size(1) + tokens_per_frame]
                    return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp

                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)
                    z_tf, z_ar = forward_predictions(h)
                    jloss = loss_fn(z_tf, h)
                    sloss = loss_fn(z_ar, h)
                    loss = jloss + sloss

                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                return (float(loss), float(jloss), float(sloss), _new_lr, _new_wd)

            (loss, jloss, sloss, _new_lr, _new_wd), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            jloss_meter.update(jloss)
            sloss_meter.update(sloss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms)
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f [%.2f, %.2f] "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            jloss_meter.avg,
                            sloss_meter.avg,
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )
                    # Log to wandb
                    if rank == 0:
                        wandb.log({
                            "train/loss": loss,
                            "train/jloss": jloss,
                            "train/sloss": sloss,
                            "train/loss_avg": loss_meter.avg,
                            "train/lr": _new_lr,
                            "train/weight_decay": _new_wd,
                            "timing/iter_ms": iter_elapsed_time_ms,
                            "timing/gpu_ms": gpu_etime_ms,
                            "timing/data_ms": data_elapsed_time_ms,
                            "epoch": epoch + 1,
                            "iteration": itr,
                        })

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        # -- End of epoch
        logger.info("avg. loss %.3f" % loss_meter.avg)

        # -- Save latest checkpoint
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)

        # -- NEW: Validation and early stopping check
        if (epoch + 1) % val_freq == 0:
            logger.info("Running validation...")
            val_loss, val_jloss, val_sloss = validate()
            logger.info(f"Validation: loss={val_loss:.4f} [jloss={val_jloss:.4f}, sloss={val_sloss:.4f}]")
            val_csv_logger.log(epoch + 1, val_loss, val_jloss, val_sloss)

            # Log validation metrics to wandb
            if rank == 0:
                wandb.log({
                    "val/loss": val_loss,
                    "val/jloss": val_jloss,
                    "val/sloss": val_sloss,
                    "val/best_loss": best_val_loss,
                    "epoch": epoch + 1,
                })

            # Check if validation improved
            if val_loss < (best_val_loss - min_delta):
                logger.info(f"★ Validation improved: {best_val_loss:.4f} → {val_loss:.4f}")
                best_val_loss = val_loss
                epochs_without_improvement = 0
                save_checkpoint(epoch + 1, best_path, val_loss=val_loss, is_best=True)
            else:
                epochs_without_improvement += val_freq
                logger.info(f"No improvement for {epochs_without_improvement} epochs (best: {best_val_loss:.4f})")

                # Early stopping check
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(f"Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
                    logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {epoch + 1 - epochs_without_improvement}")
                    logger.info(f"Best model saved to: {best_path}")
                    break

    # Finish wandb run
    if rank == 0:
        wandb.finish()

    logger.info("Training complete!")
    if epochs_without_improvement < early_stopping_patience:
        logger.info(f"Completed all {num_epochs} epochs")
    else:
        logger.info(f"Stopped early at epoch {epoch + 1}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best model: {best_path}")
    logger.info(f"Latest model: {latest_path}")


if __name__ == "__main__":
    main_with_early_stopping(args=get_args())
