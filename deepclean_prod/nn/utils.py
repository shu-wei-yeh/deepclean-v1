from ..logger import Logger
import torch
import numpy as np
import os
import csv
import time
import logging

logger = logging.getLogger(__name__)


def print_train():
    """Most important function."""
    logger.info('------------------------------------')
    logger.info('------------------------------------')
    logger.info('')
    logger.info('         ..oo0  ...ooOO00           ')
    logger.info('        ..     ...             !!!  ')
    logger.info('       ..     ...      o       \\o/  ')
    logger.info('   Y  ..     /III\\    /L ---    n   ')
    logger.info('  ||__II_____|\\_/| ___/_\\__ ___/_\\__')
    logger.info('  [[____\\_/__|/_\\|-|______|-|______|')
    logger.info(' //0 ()() ()() 0   00    00 00    00')
    logger.info('')
    logger.info('------------------------------------')
    logger.info('------------------------------------')


def get_last_checkpoint(directory, max_epochs=10000):
    """Get last checkpoint of a model from a directory."""
    checkpoint = None
    for i in range(max_epochs):
        temp = os.path.join(directory, f'epoch_{i}')
        if not os.path.exists(temp):
            return checkpoint
        checkpoint = temp


def get_device(device):
    """Convenient function to set up hardware."""
    if device.lower() == 'cpu':
        device = torch.device('cpu')
    elif 'cuda' in device.lower():
        if torch.cuda.is_available():
            device = torch.device(device)
        else:
            logging.warning('No GPU available. Use CPU instead.')
            device = torch.device('cpu')

    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory
        total_memory *= 1e-9
        logger.info('- Use device: {}'.format(
            torch.cuda.get_device_name(device)
        ))
        logger.info('- Total memory: {:.4f} GB'.format(total_memory))
    else:
        logger.info('- Use device: CPU')

    return device


def _cuda_sync(device):
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _empty_component_accumulator():
    return {
        'raw_mse': 0.0,
        'raw_psd': 0.0,
        'raw_coh': 0.0,
        'raw_tf': 0.0,
        'weighted_mse': 0.0,
        'weighted_psd': 0.0,
        'weighted_coh': 0.0,
        'weighted_tf': 0.0,
        'total': 0.0,
    }


def _accumulate_components(acc, criterion, batch_size):
    """
    Accumulate criterion component values using the same reduction convention
    as the main train/validation loss.
    """
    if not hasattr(criterion, 'latest_loss_values'):
        return

    raw = criterion.latest_loss_values
    weighted = getattr(
        criterion,
        'latest_weighted_loss_values',
        {},
    )

    scale = batch_size if criterion.reduction == 'mean' else 1.0

    for name in ('mse', 'psd', 'coh', 'tf'):
        acc[f'raw_{name}'] += float(raw.get(name, 0.0)) * scale
        acc[f'weighted_{name}'] += (
            float(weighted.get(name, 0.0)) * scale
        )

    acc['total'] += float(raw.get('total', 0.0)) * scale


def _normalize_components(acc, n_examples):
    if n_examples <= 0:
        return dict(acc)
    return {k: v / float(n_examples) for k, v in acc.items()}


def _append_component_csv(path, epoch, split, values):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        'epoch',
        'split',
        'raw_psd',
        'raw_mse',
        'raw_coh',
        'raw_tf',
        'weighted_psd',
        'weighted_mse',
        'weighted_coh',
        'weighted_tf',
        'total',
    ]

    exists = os.path.exists(path)

    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()

        row = {'epoch': epoch, 'split': split}
        row.update({k: values.get(k, 0.0) for k in fieldnames[2:]})
        writer.writerow(row)


def _write_loss_timing_csv(path, summary):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'loss',
            'calls',
            'total_seconds',
            'mean_ms_per_call',
        ])

        for name in ('psd', 'mse', 'coh', 'tf', 'criterion_total'):
            s = summary.get(name, {})
            writer.writerow([
                name,
                int(s.get('calls', 0)),
                float(s.get('total_s', 0.0)),
                float(s.get('mean_ms', 0.0)),
            ])


def _write_stage_timing_csv(path, timing):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'stage',
            'total_seconds',
            'mean_ms_per_batch',
            'calls',
        ])

        for name in (
            'model_forward',
            'loss_forward',
            'backward',
            'optimizer_step',
            'train_epoch_wall',
        ):
            total = timing[name]['total_s']
            calls = timing[name]['calls']
            mean_ms = 1000.0 * total / calls if calls > 0 else 0.0
            writer.writerow([
                name,
                total,
                mean_ms,
                calls,
            ])


def _new_stage_timing():
    return {
        'model_forward': {'total_s': 0.0, 'calls': 0},
        'loss_forward': {'total_s': 0.0, 'calls': 0},
        'backward': {'total_s': 0.0, 'calls': 0},
        'optimizer_step': {'total_s': 0.0, 'calls': 0},
        'train_epoch_wall': {'total_s': 0.0, 'calls': 0},
    }


def _add_stage_time(timing, name, elapsed):
    timing[name]['total_s'] += float(elapsed)
    timing[name]['calls'] += 1


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    lr_scheduler,
    val_loader=None,
    max_epochs=1,
    logger=None,
    device='cpu',
    early_stopping=False,
    early_stopping_patience=8,
    early_stopping_min_delta=1e-4,
    early_stopping_min_epochs=0,
):
    """
    Train network with CompositePSDLoss.

    Additional outputs
    ------------------
    <train_dir>/metrics/loss_components.csv
        Raw and weighted PSD/MSE/COH/TF values for train and validation.

    <train_dir>/metrics/loss_timing.csv
        Forward-computation time for each active loss component.

    <train_dir>/metrics/training_stage_timing.csv
        Model-forward, loss-forward, backward and optimizer-step timing.

    Notes
    -----
    - Per-loss timing comes from criterion.py and measures the LOSS FORWARD
      computation only.
    - Backward time cannot be cleanly assigned to each loss when all active
      losses are summed into one scalar and .backward() is called once.
    - Accurate CUDA profiling synchronizes the GPU and therefore adds overhead.
      Enable it only for profiling runs.
    """

    if logger is None:
        logger = Logger(
            outdir='outdir',
            label='run',
            metrics=['loss'],
        )

    num_batches = len(train_loader)
    print_train()

    early_stopping = bool(early_stopping)
    early_stopping_patience = int(early_stopping_patience)
    early_stopping_min_delta = float(early_stopping_min_delta)
    early_stopping_min_epochs = int(early_stopping_min_epochs)

    if early_stopping_patience < 1:
        raise ValueError(
            "early_stopping_patience must be >= 1"
        )

    if early_stopping_min_delta < 0:
        raise ValueError(
            "early_stopping_min_delta must be >= 0"
        )

    if early_stopping_min_epochs < 0:
        raise ValueError(
            "early_stopping_min_epochs must be >= 0"
        )

    if early_stopping and val_loader is None:
        globals()['logger'].warning(
            "Early stopping requested but val_loader is None; "
            "disabling early stopping."
        )
        early_stopping = False

    metrics_dir = os.path.join(logger.data_subdir, 'metrics')
    component_csv = os.path.join(
        metrics_dir, 'loss_components.csv'
    )
    loss_timing_csv = os.path.join(
        metrics_dir, 'loss_timing.csv'
    )
    stage_timing_csv = os.path.join(
        metrics_dir, 'training_stage_timing.csv'
    )
    early_stopping_file = os.path.join(
        metrics_dir, 'early_stopping.txt'
    )

    # Start clean for this run.
    for p in (
        component_csv,
        loss_timing_csv,
        stage_timing_csv,
        early_stopping_file,
    ):
        if os.path.exists(p):
            os.remove(p)

    if hasattr(criterion, 'reset_timing_stats'):
        criterion.reset_timing_stats()

    stage_timing = _new_stage_timing()
    profile_timing = bool(
        getattr(criterion, 'enable_timing', False)
    )

    best_val_loss = float("inf")
    best_val_epoch = None
    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch = None

    globals()['logger'].info(
        "Early stopping: enabled=%s patience=%d min_delta=%g "
        "min_epochs=%d",
        early_stopping,
        early_stopping_patience,
        early_stopping_min_delta,
        early_stopping_min_epochs,
    )

    for epoch in range(max_epochs):
        epoch_t0 = time.perf_counter()

        train_loss = 0.0
        train_components = _empty_component_accumulator()
        model.train()

        for i_batch, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(device)
            witness = data[:, :-1, :]
            target = data[:, -1, :]

            # ---------------- model forward ----------------
            if profile_timing:
                _cuda_sync(device)
                t0 = time.perf_counter()
                pred = model(witness)
                _cuda_sync(device)
                _add_stage_time(
                    stage_timing,
                    'model_forward',
                    time.perf_counter() - t0,
                )
            else:
                pred = model(witness)

            # ---------------- loss forward -----------------
            if profile_timing:
                _cuda_sync(device)
                t0 = time.perf_counter()
                loss = criterion(pred, target, witness)
                _cuda_sync(device)
                _add_stage_time(
                    stage_timing,
                    'loss_forward',
                    time.perf_counter() - t0,
                )
            else:
                loss = criterion(pred, target, witness)

            _accumulate_components(
                train_components,
                criterion,
                len(data),
            )

            # ---------------- backward ---------------------
            if profile_timing:
                _cuda_sync(device)
                t0 = time.perf_counter()
                loss.backward()
                _cuda_sync(device)
                _add_stage_time(
                    stage_timing,
                    'backward',
                    time.perf_counter() - t0,
                )
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            # ---------------- optimizer --------------------
            if profile_timing:
                _cuda_sync(device)
                t0 = time.perf_counter()
                optimizer.step()
                _cuda_sync(device)
                _add_stage_time(
                    stage_timing,
                    'optimizer_step',
                    time.perf_counter() - t0,
                )
            else:
                optimizer.step()

            if criterion.reduction == 'mean':
                train_loss += loss.item() * len(data)
            else:
                train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        train_components = _normalize_components(
            train_components,
            len(train_loader.dataset),
        )

        _append_component_csv(
            component_csv,
            epoch,
            'train',
            train_components,
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        val_loss = 0.0
        val_components = _empty_component_accumulator()

        if val_loader is not None:
            model.eval()

            # Keep loss timing focused on TRAINING calls only.
            timing_state = getattr(
                criterion,
                'enable_timing',
                False,
            )
            if hasattr(criterion, 'enable_timing'):
                criterion.enable_timing = False

            with torch.no_grad():
                for i_batch, (data, _) in enumerate(val_loader):
                    data = data.to(device)
                    witness = data[:, :-1, :]
                    target = data[:, -1, :]

                    pred = model(witness)
                    loss = criterion(pred, target, witness)

                    _accumulate_components(
                        val_components,
                        criterion,
                        len(data),
                    )

                    if criterion.reduction == 'mean':
                        val_loss += loss.item() * len(data)
                    else:
                        val_loss += loss.item()

            if hasattr(criterion, 'enable_timing'):
                criterion.enable_timing = timing_state

            val_loss /= len(val_loader.dataset)
            val_components = _normalize_components(
                val_components,
                len(val_loader.dataset),
            )

            _append_component_csv(
                component_csv,
                epoch,
                'validation',
                val_components,
            )

        if lr_scheduler is not None:
            lr_scheduler.step()

        logger.update_metric(
            train_loss,
            val_loss,
            'loss',
            epoch,
            num_batches,
            num_batches,
        )
        logger.display_status(
            epoch,
            max_epochs,
            num_batches,
            num_batches,
            train_loss,
            val_loss,
            'loss',
        )
        logger.log_metric()
        logger.save_model(model, epoch)

        epoch_elapsed = time.perf_counter() - epoch_t0
        _add_stage_time(
            stage_timing,
            'train_epoch_wall',
            epoch_elapsed,
        )

        # Log useful component information.
        logger_msg = (
            f"Epoch {epoch}: "
            f"raw_psd={train_components['raw_psd']:.6e}, "
            f"raw_mse={train_components['raw_mse']:.6e}, "
            f"raw_coh={train_components['raw_coh']:.6e}, "
            f"raw_tf={train_components['raw_tf']:.6e}; "
            f"weighted_psd={train_components['weighted_psd']:.6e}, "
            f"weighted_mse={train_components['weighted_mse']:.6e}, "
            f"weighted_coh={train_components['weighted_coh']:.6e}, "
            f"weighted_tf={train_components['weighted_tf']:.6e}"
        )
        globals()['logger'].info(logger_msg)

        # --------------------------------------------------
        # Early stopping -- epoch-level, after validation
        # --------------------------------------------------
        if early_stopping and val_loader is not None:
            current_val_loss = float(val_loss)

            if (
                np.isfinite(current_val_loss)
                and current_val_loss
                < best_val_loss - early_stopping_min_delta
            ):
                best_val_loss = current_val_loss
                best_val_epoch = epoch
                epochs_without_improvement = 0
                globals()['logger'].info(
                    "Early stopping: validation improved at epoch %d: "
                    "val_loss=%.12g",
                    epoch + 1,
                    current_val_loss,
                )
            else:
                epochs_without_improvement += 1
                globals()['logger'].info(
                    "Early stopping: no improvement at epoch %d "
                    "(%d/%d), val_loss=%.12g, best=%.12g",
                    epoch + 1,
                    epochs_without_improvement,
                    early_stopping_patience,
                    current_val_loss,
                    best_val_loss,
                )

            completed_epochs = epoch + 1

            if (
                completed_epochs >= early_stopping_min_epochs
                and epochs_without_improvement
                >= early_stopping_patience
            ):
                stopped_early = True
                stop_epoch = epoch
                globals()['logger'].info(
                    "EARLY STOPPING triggered after epoch %d. "
                    "Best validation loss %.12g at epoch %s.",
                    completed_epochs,
                    best_val_loss,
                    (
                        str(best_val_epoch + 1)
                        if best_val_epoch is not None
                        else "N/A"
                    ),
                )
                break

    try:
        logger.plot_loss_components_epoch()

    except Exception as exc:
        logging.warning(
            "Could not create "
            "loss_components_epoch plot: %s",
            exc,
        )

    # ------------------------------------------------------
    # Early-stopping summary
    # ------------------------------------------------------
    with open(early_stopping_file, 'w') as f:
        f.write(f"enabled={early_stopping}\n")
        f.write(f"patience={early_stopping_patience}\n")
        f.write(f"min_delta={early_stopping_min_delta}\n")
        f.write(f"min_epochs={early_stopping_min_epochs}\n")
        f.write(f"stopped_early={stopped_early}\n")
        f.write(
            "stop_epoch="
            + (
                str(stop_epoch + 1)
                if stop_epoch is not None
                else "None"
            )
            + "\n"
        )
        f.write(
            "best_val_epoch="
            + (
                str(best_val_epoch + 1)
                if best_val_epoch is not None
                else "None"
            )
            + "\n"
        )
        f.write(f"best_val_loss={best_val_loss}\n")

    # ------------------------------------------------------
    # End-of-run timing summaries
    # ------------------------------------------------------
    if hasattr(criterion, 'get_timing_summary'):
        timing_summary = criterion.get_timing_summary()
        _write_loss_timing_csv(
            loss_timing_csv,
            timing_summary,
        )

        if hasattr(criterion, 'format_timing_summary'):
            globals()['logger'].info(
                '\n' + criterion.format_timing_summary()
            )

    if profile_timing:
        _write_stage_timing_csv(
            stage_timing_csv,
            stage_timing,
        )

    globals()['logger'].info(
        'Training diagnostics:\n'
        f'  components : {component_csv}\n'
        f'  early stop : {early_stopping_file}\n'
        + (
            f'  loss timing: {loss_timing_csv}\n'
            f'  stage timing: {stage_timing_csv}'
            if profile_timing
            else '  timing     : disabled'
        )
    )


def evaluate(
    dataloader,
    model,
    criterion=None,
    device='cpu',
):
    """Evaluate model and return predictions and optional loss."""

    model.eval()
    eval_loss = 0.0
    prediction = []

    with torch.no_grad():
        for i_batch, (data, _) in enumerate(dataloader):
            data = data.to(device)
            witness = data[:, :-1, :]
            target = data[:, -1, :]

            pred = model(witness)
            prediction.append(pred.cpu().numpy())

            if criterion is not None:
                loss = criterion(pred, target, witness)

                if criterion.reduction == 'mean':
                    eval_loss += loss.item() * len(data)
                else:
                    eval_loss += loss.item()

    prediction = np.concatenate(prediction)

    if criterion is not None:
        eval_loss /= len(dataloader.dataset)

    return (
        (prediction, eval_loss)
        if criterion is not None
        else prediction
    )
