#train.py
import os, sys, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

class Trainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        set_seed(config.training.seed)
        self.device = torch.device(
            config.training.device if torch.cuda.is_available() else "cpu"
        )
        print(f"[Trainer] device = {self.device}")
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        self.model = self._build_model().to(self.device)
        self.optimizer  = self._build_optimizer()
        self.criterion  = get_loss_function(
            config.training.loss_type,
            focal_alpha=config.training.focal_alpha,
            focal_gamma=config.training.focal_gamma,
            label_smoothing=config.training.label_smoothing,
        )
        self.scheduler  = self._build_scheduler()
        self.metrics_calc = MetricsCalculator()
        self.early_stopping: Optional[EarlyStopping] = None
        if config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.training.patience,
                min_delta=config.training.min_delta,
                mode="max",
            )

        self.writer = None
        print(f"[Trainer] parameters = {count_parameters(self.model):,}")

    # call before each independent train run
    def _reset(self):
        self.model     = self._build_model().to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        if self.config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.config.training.patience,
                min_delta=self.config.training.min_delta,
                mode="max",
            )

    def _build_model(self) -> nn.Module:
        # Note: adjacency_matrix parameter is used for both topology adjacency
        # and correlation-based connectivity depending on bias_type.
        # This is intentional but can be confusing - the matrix content matters,
        # not the parameter name. For bias_type='correlation', you should pass
        # a correlation matrix via the adjacency_matrix parameter.
        return SpatialSpectralGraphTransformer(
            num_channels            = self.config.data.num_channels,
            num_bands               = self.config.data.num_bands,
            num_classes             = self.config.model.num_classes,
            hidden_dim              = self.config.model.hidden_dim,
            num_spatial_layers      = self.config.model.num_spatial_layers,
            spatial_heads           = self.config.model.num_attention_heads,
            spatial_dropout         = self.config.model.spatial_dropout,
            use_positional_bias     = self.config.model.use_positional_bias,
            bias_type               = self.config.model.bias_type,
            pooling_method          = self.config.model.pooling_method,
            num_temporal_layers     = self.config.model.num_temporal_layers,
            temporal_heads          = self.config.model.temporal_heads,
            temporal_dropout        = self.config.model.temporal_dropout,
            temporal_sequence_length= self.config.model.temporal_sequence_length,
            ffn_hidden_dim          = self.config.model.ffn_hidden_dim,
            ffn_dropout             = self.config.model.ffn_dropout,
            activation              = self.config.model.activation,
            use_layer_norm          = self.config.model.layer_norm,
            adjacency_matrix        = NORMALIZED_ADJACENCY,
            distance_matrix         = DISTANCE_MATRIX,
        )

    def _build_optimizer(self) -> optim.Optimizer:
        P = dict(params=self.model.parameters(),
                 lr=self.config.training.learning_rate,
                 weight_decay=self.config.training.weight_decay)
        name = self.config.training.optimizer
        if name == "adam":   return optim.Adam(**P)
        if name == "adamw":  return optim.AdamW(**P)
        if name == "sgd":    return optim.SGD(**P, momentum=0.9)
        raise ValueError(f"Unknown optimizer: {name}")

    def _build_scheduler(self):
        name = self.config.training.scheduler
        if name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.num_epochs, eta_min=1e-6)
        if name == "step":
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        if name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=10)
        if name == "none":   return None
        raise ValueError(f"Unknown scheduler: {name}")

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_m       = AverageMeter()
        preds, tgts  = [], []

        for idx, (x, y) in enumerate(tqdm(loader, desc=f"Ep {epoch} [Train]")):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out  = self.model(x)
            loss = self.criterion(out["logits"], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training.grad_clip_norm
            )
            self.optimizer.step()

            loss_m.update(loss.item(), x.size(0))
            preds.append(out["logits"].argmax(1).cpu().numpy())
            tgts.append(y.cpu().numpy())

        m = self.metrics_calc.calculate_metrics(
            np.concatenate(preds), np.concatenate(tgts),
            self.config.model.num_classes)
        m["loss"] = loss_m.avg
        return m

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, epoch: int, tag: str = "Val") -> Dict[str, float]:
        self.model.eval()
        loss_m       = AverageMeter()
        preds, tgts  = [], []

        for x, y in tqdm(loader, desc=f"Ep {epoch} [{tag}]"):
            x, y = x.to(self.device), y.to(self.device)
            out  = self.model(x)
            loss_m.update(self.criterion(out["logits"], y).item(), x.size(0))
            preds.append(out["logits"].argmax(1).cpu().numpy())
            tgts.append(y.cpu().numpy())

        m = self.metrics_calc.calculate_metrics(
            np.concatenate(preds), np.concatenate(tgts),
            self.config.model.num_classes)
        m["loss"] = loss_m.avg
        return m

    #train→val→test run
    def run(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        test_loader:  DataLoader,
        tag: str = "run",
    ) -> Dict[str, Dict]:
        """
        Train until convergence (or early stop).
        Load best-val checkpoint, then evaluate on test.
        Returns  {"train": …, "val": …, "test": …}  from the best-val epoch.
        """
        best_val_acc   = -1.0  # Init to -1 so epoch 1 always updates
        best_train_m   = {}
        best_val_m     = {}
        ckpt_path      = os.path.join(self.config.training.checkpoint_dir, f"best_{tag}.pt")

        # Create per-run TensorBoard writer to avoid mixing logs
        local_writer = None
        if self.config.training.use_tensorboard and _HAS_TB:
            log_dir = os.path.join("/kaggle/working/runs", self.config.experiment_name, tag)
            os.makedirs(log_dir, exist_ok=True)
            local_writer = SummaryWriter(log_dir=log_dir)
            print(f"  [TensorBoard] Logging to {log_dir}")

        try:
            for epoch in range(1, self.config.training.num_epochs + 1):
                tr_m  = self.train_epoch(loader=train_loader, epoch=epoch)
                val_m = self.evaluate(loader=val_loader, epoch=epoch, tag="Val")

                print(f"  Ep {epoch:3d} | train loss={tr_m['loss']:.4f} acc={tr_m['accuracy']:.4f}" +
                      f"  |  val loss={val_m['loss']:.4f} acc={val_m['accuracy']:.4f}")

                # Scheduler step (do this BEFORE logging LR for correct value)
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_m["accuracy"])
                    else:
                        self.scheduler.step()

                # TensorBoard logging (log LR AFTER scheduler step)
                if local_writer:
                    for prefix, mm in [("Train", tr_m), ("Val", val_m)]:
                        local_writer.add_scalar(f"{prefix}/Loss",     mm["loss"],     epoch)
                        local_writer.add_scalar(f"{prefix}/Accuracy", mm["accuracy"], epoch)
                    # Log current LR (after scheduler step)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    local_writer.add_scalar("LR", current_lr, epoch)

                # Best checkpoint
                if val_m["accuracy"] > best_val_acc:
                    best_val_acc = val_m["accuracy"]
                    best_train_m = tr_m
                    best_val_m   = val_m
                    if self.config.training.save_best_only:
                        save_checkpoint(
                            {"epoch": epoch,
                             "model_state_dict": self.model.state_dict(),
                             "optimizer_state_dict": self.optimizer.state_dict(),
                             "best_val_acc": best_val_acc},
                            ckpt_path)

                # Early stop
                if self.early_stopping and self.early_stopping(val_m["accuracy"]):
                    print(f"  Early stopping at epoch {epoch}")
                    break

        finally:
            # Always close writer even if exception occurs
            if local_writer:
                local_writer.close()

        # ── test with best weights ──
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location=self.device)["model_state_dict"])
        test_m = self.evaluate(loader=test_loader, epoch=0, tag="Test")

        print(f"  >>> best_val_acc={best_val_acc:.4f}   test_acc={test_m['accuracy']:.4f}")
        return {"train": best_train_m, "val": best_val_m, "test": test_m}

def _make_factory(cfg: ExperimentConfig) -> SeedLoaderFactory:
    return SeedLoaderFactory(
        use_mat               = cfg.data.use_mat,
        mat_dir               = cfg.data.seed_eeg_path,
        label_mat             = cfg.data.label_mat_path,
        npz_dir               = cfg.data.seed_multimodal_path,
        feature_key_prefix    = cfg.data.feature_key_prefix,
        normalization         = cfg.data.normalize,
        normalization_method  = cfg.data.normalization_method,
        window_size           = cfg.data.window_size,
        stride                = cfg.data.stride,
    )

def run_within_subject(config: ExperimentConfig) -> Dict:
    """
    Loop: for every subject × every session  →  clip-split train/val/test.
    """
    factory = _make_factory(config)
    trainer = Trainer(config)

    num_subjects = 15 if config.data.use_mat else 12
    results: Dict = {}

    for sid in range(1, num_subjects + 1):
        for sess in (1, 2, 3):
            tag = f"s{sid}_sess{sess}"
            print(f"\n{'='*60}\n  Within-subject | Subject {sid} | Session {sess}\n{'='*60}")

            trainer._reset()   # fresh weights for every (subject, session)

            train_loader, val_loader, test_loader = factory.within_subject_loaders(
                subject_id  = sid,
                session_idx = sess,
                val_clip    = config.split.val_clip,
                batch_size  = config.training.batch_size,
                num_workers = config.training.num_workers,
            )

            results[tag] = trainer.run(train_loader, val_loader, test_loader, tag=tag)

    # summary
    test_accs = [r["test"]["accuracy"] for r in results.values()]
    results["_summary"] = {
        "mean_test_acc": float(np.mean(test_accs)),
        "std_test_acc":  float(np.std(test_accs)),
    }
    print(f"\n{'='*60}")
    print(f"  WITHIN-SUBJECT SUMMARY  "
          f"mean={results['_summary']['mean_test_acc']:.4f} "
          f"± {results['_summary']['std_test_acc']:.4f}")
    print(f"{'='*60}")

    _save_json(config, results, "within_subject")
    return results

def run_cross_subject(config: ExperimentConfig) -> Dict:
    """
    Cross-subject evaluation.
    Can use session 1 only (default) or all 3 sessions based on config.
    """
    factory = _make_factory(config)
    trainer = Trainer(config)

    print(f"\n{'='*60}\n  Cross-subject\n{'='*60}")
    print(f"  train = {config.split.train_subjects}")
    print(f"  val   = {config.split.val_subjects}")
    print(f"  test  = {config.split.test_subjects}")
    print(f"  sessions = {config.training.cross_subject_sessions}")

    if config.training.cross_subject_sessions == 'first':
        # Use session 1 only (faster, standard)
        train_loader, val_loader, test_loader = factory.cross_subject_loaders(
            train_subjects = config.split.train_subjects,
            val_subjects   = config.split.val_subjects,
            test_subjects  = config.split.test_subjects,
            session_idx    = 1,
            batch_size     = config.training.batch_size,
            num_workers    = config.training.num_workers,
        )
        result = trainer.run(train_loader, val_loader, test_loader, tag="cross")
        
    else:  # 'all' - use all 3 sessions
        # Train on all 3 sessions, report average performance
        all_results = []
        for sess in (1, 2, 3):
            print(f"\n  --- Session {sess} ---")
            trainer._reset()  # Fresh weights for each session
            
            train_loader, val_loader, test_loader = factory.cross_subject_loaders(
                train_subjects = config.split.train_subjects,
                val_subjects   = config.split.val_subjects,
                test_subjects  = config.split.test_subjects,
                session_idx    = sess,
                batch_size     = config.training.batch_size,
                num_workers    = config.training.num_workers,
            )
            sess_result = trainer.run(train_loader, val_loader, test_loader, tag=f"cross_s{sess}")
            all_results.append(sess_result)
        
        # Average results across sessions
        result = {
            "train": {k: np.mean([r["train"][k] for r in all_results]) for k in all_results[0]["train"]},
            "val":   {k: np.mean([r["val"][k] for r in all_results])   for k in all_results[0]["val"]},
            "test":  {k: np.mean([r["test"][k] for r in all_results])  for k in all_results[0]["test"]},
            "session_results": all_results,  # Keep individual session results
        }
        print(f"\n  Average across 3 sessions:")
        print(f"    test_acc = {result['test']['accuracy']:.4f}")

    _save_json(config, result, "cross_subject")
    return result

def _save_json(config: ExperimentConfig, data: Dict, name: str):
    def _convert(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        return o

    path = os.path.join(config.training.checkpoint_dir, f"results_{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, default=_convert, indent=2)
    print(f"  [saved] {path}")

def main():
    config = get_seed_within_config()          # ← swap to get_seed_cross_config() if needed

    if config.split.mode == "within":
        run_within_subject(config)
    elif config.split.mode == "cross":
        run_cross_subject(config)
    else:
        raise ValueError(f"Unknown split.mode = {config.split.mode}")


if __name__ == "__main__":
    main()