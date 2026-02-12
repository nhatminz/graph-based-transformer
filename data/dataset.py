#dataset.py
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, MinMaxScaler


BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

@dataclass
class Trial: 
    # X: (L, 62,5) where L = number of 1s steps in that clip
    # y: (0,1,2)
    x: np.ndarray
    y: int
    meta: Dict

def map_label_seed(y_raw: int) -> int:
    return int(y_raw) + 1 # -1,0,1 -> 0,1,2

class SeedMatReader:
    #dataset for .mat file, 15 subjects
    def __init__(self, mat_dir: str, label_mat_path: str, feature_key_prefix: str = "de_LDS"): # or can be "de_movingAve"
        self.mat_dir = mat_dir
        self.label_mat_path = label_mat_path
        self.feature_key_prefix = feature_key_prefix
        self.labels_15 = self._load_labels_15()

    def _load_label_15(self) -> np.ndarray:
        lab = loadmat(self.label_mat_path)
        if "label" not in lab:
            raise KeyError("label.mat must contain key 'label'")
        labels = np.array(lab['label']).reshape(-1) # (15,)
        if labels.size != 15:
            raise ValueError(f"Expected 15 labels, got {labels.size}")
        return labels.astype(int)

    def list_subject_files(self, subject_id: int) -> List[str]:
        files = [
            f for f in os.listdir(self.mat_dir)
            if f.endswith(".mat") and f.split("_")[0].isdigit()
            and int(f.split("_")[0]) == subject_id
        ]
        files.sort()
        return files

    def load_subject_session_trials(
        self,
        subject_id: int,
        session_idx: int,
        clips: Optional[List[int]] = None, # subset of 1..15
    ) -> List[Trial]:
        files = self.list_subject_files(subject_id)
        if not (1 <= session_idx <= len(files)):
            raise ValueError(f"Subject {subject_id} session {session_idx} not found. Available={len(files)}")

        path = os.path.join(self.mat_dir, files[session_idx - 1])
        mat = loatmat(path)

        clip_ids = clips if lips is not None else list(range(1,16))
        trials: List[Trial] = []

        for clip_id in clip_ids:
            key = f"{self.feature_key_prefix}{clip_id}"
            if key not in mat:
                raise KeyError(f"Missing key '{key}' in {path}")

            arr = np.array(mat[key]) # (62, L, 5)
            arr = squeeze(arr)
            if arr.ndim != 3:
                raise ValueError(f"{key} expected 3D (62,L,5), got {arr.shape}")
            # Ensure shape = (L,62,5)
            if arr.shape[0] != 62 and arr.shape[1] == 62:
                # (L,62,5) already
                x = arr.astype(np.float32)
            else:
                # expected (62,L,5) -> (L,62,5)
                x = np.transpose(arr, (1, 0, 2)).astype(np.float32)
            y_raw = int(self.labels_15[clip_id - 1])
            y = map_label_seed(y_raw)
            
            trials.append(Trial(
                x=x, y=y,
                meta={"subject": subject_id, "session": session_idx, "clip": clip_id, "source": os.path.basename(path)}
            ))
        return trials

class SeedNpzReader:
    #dataset for .npz, 12 subjects
    def __init__(self, npz_dir: str, band_order: List[str] = BANDS):
        self.npz_dir = npz_dir
        self.band_order = band_order

    def _load_npz(self, subject_id: int, session_idx: int) -> Dict:
        path = os.path.join(self.npz_dir, f"{subject_id}_{session_idx}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return np.load(path, allow_pickle=True)
        
    def _stack_bands(self, band_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # each band is (N,62) -> stack -> (N,62,5)
        xs = [band_dict[b].astype(np.float32) for b in self.band_order]
        return np.stack(xs, axis=-1)

    def _segment_by_label_runs(self, x: np.ndarray, y: np.ndarray) -> List[Trial]:
        """
        x: (N,62,5), y: (N,)
        We split contiguous segments where label stays constant.
        In ideal case, should produce 9 segments for train split and 6 for test split.
        """
        y = y.astype(int)
        trials: List[Trial] = []
        start = 0
        for i in range(1, len(y) + 1):
            if i == len(y) or y[i] != y[i - 1]:
                seg_x = x[start:i]
                seg_y = int(y[i - 1]) # npz was labeled with 0,1,2 already
                if len(seg_x) > 0:
                    trials.append(Trial(seg_x, seg_y, meta={"clip_like": len(trials) + 1}))
                start = i
        return trials

    def load_subject_session_trials(self, subject_id: int, session_idx: int) -> Tuple[List[Trial], List[Trial]]:
        npz = self._load_npz(subject_id, session_idx)

        train_dict = pickle.loads(npz["train_data"].item())
        test_dict = pickle.loads(npz["test_data"].item())
        y_train = npz["train_label"].astype(int)
        y_test  = npz["test_label"].astype(int)

        x_train = self._stack_bands(train_dict)  # (Ntrain,62,5)
        x_test  = self._stack_bands(test_dict)   # (Ntest,62,5)

        train_trials = self._segment_by_label_runs(x_train, y_train)  # ideally 9 trials
        test_trials  = self._segment_by_label_runs(x_test, y_test)    # ideally 6 trials

        # add metadata
        for t in train_trials:
            t.meta.update({"subject": subject_id, "session": session_idx, "split": "train"})
        for t in test_trials:
            t.meta.update({"subject": subject_id, "session": session_idx, "split": "test"})

        return train_trials, test_trials

@dataclass
class WindowIndex:
    trial_idx: int
    start: int

class SeedWindowDataset(Dataset):
    #returns: x_window: (t, 62, 5), y
    def __init__(self, trials: List[Trial], window_size: int, stride: int):
        self.trials = trials
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.index: List[WindowIndex] = []
        self._build_index()

    def _build_index(self):
        self.index.clear()
        for tid, tr in enumerate(self.trials):
            L = tr.x.shape[0]
            if L < self.window_size:
                continue
            for s in range(0, L - self.window_size + 1, self.stride):
                self.index.append(WindowIndex(trial_idx=tid, start=s))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        wi = self.index[idx]
        tr = self.trials[wi.trial_idx]
        x = tr.x[wi.start:wi.start + self.window_size] # (T, 62, 5)
        y = tr.y
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)

def build_scaler(method: Literal["zscore", "minmax"] = "zscore"):
    return StandardScaler() if method == "zscore" else MinMaxScaler()

def fit_scaler_on_trials(trials: List[Trial], method: Literal["zscore", "minmax"] = "zscore") -> object:
    """
    Fit scaler only on TRAIN trials.
    We normalize per-feature (channel, band) by flattening (L,62,5)->(L,310) over all L.
    """
    scaler = build_scaler(method)
    # gather all time steps across train trials
    chunks = []
    for tr in trials:
        x = tr.x  # (L,62,5)
        flat = x.reshape(x.shape[0], -1)  # (L,310)
        chunks.append(flat)
    X = np.concatenate(chunks, axis=0)  # (TotalL,310)

    scaler.fit(X)
    return scaler

def apply_scaler_to_trials(trials: List[Trial], scaler: object) -> List[Trial]:
    out: List[Trial] = []
    for tr in trials:
        x = tr.x
        flat = x.reshape(x.shape[0], -1)
        flat = scaler.transform(flat)
        x2 = flat.reshape(x.shape[0], 62, 5).astype(np.float32)
        out.append(Trial(x=x2, y=tr.y, meta=dict(tr.meta)))
    return out

class SeedLoaderFactory:
    def __init__(
        self,
        *,
        use_mat: bool,
        mat_dir: Optional[str] = None,
        label_mat: Optional[str] = None,
        npz_dir: Optional[str] = None,
        feature_key_prefix: str = "de_LDS",
        normalization: bool = True,
        normalization_method: Literal["zscore", "minmax"] = "zscore",
        window_size: int = 1,
        stride: int =1,
    ):
        self.use_mat = use_mat
        self.normalization = normalization
        self.normalization_method = normalization_method
        self.window_size = window_size
        self.stride = stride

        if use_mat:
            if mat_dir is None or label_mat is None:
                raise ValueError("use_mat=True requires mat_dir and label_mat")
            self.mat_reader = SeedMatReader(mat_dir, label_mat, feature_key_prefix)
            self.npz_reader = None
        else:
            if npz_dir is None:
                raise ValueError("use_mat=False requires npz_dir")
            self.npz_reader = SeedNpzReader(npz_dir)
            self.mat_reader = None

    # ---------- Mode A: Within-subject, clip split ----------
    def within_subject_loaders(
        self,
        subject_id: int,
        session_idx: int,
        *,
        val_clip: int = 1,
        batch_size: int = 64,
        num_workers: int = 2,
        pin_memory: Optional[bool] = None,
    ) -> Tuple[Dataloader, Dataloader, Dataloader]:
        """
        Train: clips 1..9 except val_clip
        Val:   val_clip (within 1..9)
        Test:  clips 10..15
        """
        #Load trials
        if self.use_mat:
            train_clips = [c for c in range(1,10) if c != val_clip]
            val_clips = [val_clip]
            test_clips = list(range(10,16))

            train_trials = self.mat_reader.load_subject_session_trials(subject_id, session_idx, train_clips)
            val_trials   = self.mat_reader.load_subject_session_trials(subject_id, session_idx, val_clips)
            test_trials  = self.mat_reader.load_subject_session_trials(subject_id, session_idx, test_clips)
        else: 
            # NPZ already gives split 9/6 (clip-like segments by label runs)
            tr_trials, te_trials = self.npz_reader.load_subject_session_trials(subject_id, session_idx)

            #pick val clip among first 9 clips
            if not(1 <= val_clip <= len(tr_trials)):
                raise ValueError(f"val_clip={val_clip} out of range for npz train segments={len(tr_trials)}")

            val_trials = [tr_trials[val_clip - 1]]
            train_trials = [t for i, t in enumerate(tr_trials) if i != (val_clip - 1)]
            test_trials = te_trials

        #fit scaler on train only, then apply to val/test
        if self.normalization:
            scaler = fit_scaler_on_trials(train_trials, self.normalization_method)
            train_trials = apply_scaler_to_trials(train_trials, scaler)
            val_trials   = apply_scaler_to_trials(val_trials, scaler)
            test_trials  = apply_scaler_to_trials(test_trials, scaler)

        train_ds = SeedWindowDataset(train_trials, self.window_size, self.stride)
        val_ds   = SeedWindowDataset(val_trials,   self.window_size, self.stride)
        test_ds  = SeedWindowDataset(test_trials,  self.window_size, self.stride)

        if pin_memory is None:
            pin_memory = torch.cuda.is_available()

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=True, persistent_workers=(num_workers > 0)
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=False, persistent_workers=(num_workers > 0)
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=False, persistent_workers=(num_workers > 0)
        )

        return train_loader, val_loader, test_loader

    # ---------- Mode B: Cross-subject ----------
    def cross_subject_loaders(
        self,
        *,
        train_subjects: List[int],
        val_subjects: List[int],
        test_subjects: List[int],
        session_idx: int,
        batch_size = 64,
        num_workers: int = 2,
        pin_memory: Optional[bool] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Cross-subject split.
        For each subject, use the same within-subject protocol:
          train clips 1..9
          test  clips 10..15
        BUT in cross-subject we typically:
          - train set: subjects in train_subjects using clips 1..9
          - val set:   subjects in val_subjects using clips 1..9 (or a fixed val_clip per subject if you want)
          - test set:  subjects in test_subjects using clips 10..15 (common)
        """
        def load_trials_for(subjects: List[int], clips: List[int], split_name: str) -> List[Trial]:
            all_trials: List[trial] = []
            for sid in subjects:
                if self.use_mat:
                    ts = self.mat_reader.load_subject_session_trials(sid, session_idx, clips)
                else:
                    tr_trials, te_trials = self.npz_reader.load_subject_session_trials(sid, session_idx)
                    # for NPZ: clips 1..9 correspond to tr_trials, clips 10..15 correspond to te_trials
                    if clips == list(range(1, 10)):
                        ts = tr_trials
                    elif clips == list(range(10, 16)):
                        ts = te_trials
                    else:
                        raise ValueError("NPZ cross-subject currently supports only clips 1..9 or 10..15 sets.")
                # tag meta
                for t in ts:
                    t.meta["split"] = split_name
                all_trials.extend(ts)
            return all_trials

        train_clips = list(range(1, 10))
        test_clips  = list(range(10, 16))

        train_trials = load_trials_for(train_subjects, train_clips, "train")
        val_trials   = load_trials_for(val_subjects,   train_clips, "val")
        test_trials  = load_trials_for(test_subjects,  test_clips,  "test")

        # Normalize: fit only on train_trials
        if self.normalization:
            scaler = fit_scaler_on_trials(train_trials, self.normalization_method)
            train_trials = apply_scaler_to_trials(train_trials, scaler)
            val_trials   = apply_scaler_to_trials(val_trials, scaler)
            test_trials  = apply_scaler_to_trials(test_trials, scaler)

        train_ds = SeedWindowDataset(train_trials, self.window_size, self.stride)
        val_ds   = SeedWindowDataset(val_trials,   self.window_size, self.stride)
        test_ds  = SeedWindowDataset(test_trials,  self.window_size, self.stride)

        if pin_memory is None:
            pin_memory = torch.cuda.is_available()

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=True, persistent_workers=(num_workers > 0)
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=False, persistent_workers=(num_workers > 0)
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=False, persistent_workers=(num_workers > 0)
        )

        return train_loader, val_loader, test_loader