import os
import glob
import random
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from monai.data import DataLoader, CacheDataset
from monai.transforms import Compose

from transforms import get_train_transforms, get_test_val_transforms, get_inference_transforms

logger = logging.getLogger("BraTS2025-Dataset")
logging.basicConfig(level=logging.INFO)

MODALITIES = ['t1n', 't1c', 't2w', 't2f']  # Naming as per your structure
LABEL_KEY = 'seg'
CASE_SUFFIX = {
    't1n': '-t1n.nii.gz',
    't1c': '-t1c.nii.gz',
    't2w': '-t2w.nii.gz',
    't2f': '-t2f.nii.gz',
    'seg': '-seg.nii.gz'
}

def find_all_patients(data_dir: str) -> Dict[str, List[str]]:
    """
    Traverse the Training directory and return a mapping from patient ID to their timepoints (case subdirs).
    Debug printout included.
    """
    pattern = os.path.join(data_dir, '*')
    cases = [os.path.basename(d) for d in glob.glob(pattern) if os.path.isdir(d)]
    print(f"DEBUG: Found {len(cases)} case folders in {data_dir}")
    print(f"DEBUG: Example cases: {cases[:10]}")
    patient_to_cases = {}
    for case in cases:
        try:
            parts = case.split('-')
            patient_id = '-'.join(parts[:3])
            if patient_id not in patient_to_cases:
                patient_to_cases[patient_id] = []
            patient_to_cases[patient_id].append(case)
        except Exception as e:
            logger.warning(f"Could not parse case name: {case}. Error: {e}")
    print(f"DEBUG: Found {len(patient_to_cases)} unique patients.")
    for pid, clist in list(patient_to_cases.items())[:5]:
        print(f"DEBUG: Patient {pid} has timepoints: {clist}")
    return patient_to_cases

def patient_wise_split(patient_to_cases: Dict[str, List[str]], val_pct=0.15, test_pct=0.15, seed=42):
    """
    Split the patients into train, val, test splits. All timepoints for a patient go into the same split.
    Debug printout included.
    """
    patient_ids = list(patient_to_cases.keys())
    random.seed(seed)
    random.shuffle(patient_ids)
    n = len(patient_ids)
    n_val = int(round(val_pct * n))
    n_test = int(round(test_pct * n))
    n_train = n - n_val - n_test
    print(f"DEBUG: n={n}, n_train={n_train}, n_val={n_val}, n_test={n_test}")
    splits = {'train': [], 'val': [], 'test': []}
    splits['train'] = sum([patient_to_cases[pid] for pid in patient_ids[:n_train]], [])
    splits['val'] = sum([patient_to_cases[pid] for pid in patient_ids[n_train:n_train + n_val]], [])
    splits['test'] = sum([patient_to_cases[pid] for pid in patient_ids[n_train + n_val:]], [])
    print(f"DEBUG: Split sizes. Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    logger.info(f"Dataset split: {n_train} train, {n_val} val, {n_test} test patients.")
    return splits

def create_case_dicts(data_dir: str, cases: List[str], has_label: bool = True) -> List[Dict]:
    """
    For each case, create a dict with keys: 'image' (a list of modality paths), 'label' (seg path).
    Debug printout included.
    """
    data_list = []
    missing_cases = []
    for case in cases:
        case_dir = os.path.join(data_dir, case)
        entry = {}
        try:
            entry['image'] = [os.path.join(case_dir, f"{case}{CASE_SUFFIX[m]}") for m in MODALITIES]
            if has_label:
                entry['label'] = os.path.join(case_dir, f"{case}{CASE_SUFFIX[LABEL_KEY]}")
            entry['case_id'] = case
            # Check existence of all files
            all_exist = all([os.path.exists(p) for p in entry['image']])
            if has_label:
                all_exist = all_exist and os.path.exists(entry['label'])
            if not all_exist:
                missing_cases.append(case)
                logger.warning(f"Missing files for case {case}, skipping.")
                continue
            data_list.append(entry)
        except Exception as e:
            logger.error(f"Error processing case {case}: {e}")
    print(f"DEBUG: Valid cases loaded: {len(data_list)} (out of {len(cases)})")
    if missing_cases:
        print(f"DEBUG: Cases skipped due to missing files: {missing_cases[:5]} ... total {len(missing_cases)}")
    return data_list

class BraTSDataset(Dataset):
    """
    Custom MONAI-compatible dataset for BraTS 2025.
    """
    def __init__(self, data_list: List[Dict], transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        sample = {
            "image": data['image'],
            "case_id": data.get('case_id', None)
        }
        if 'label' in data:
            sample['label'] = data['label']
        if self.transform:
            sample = self.transform(sample)
        return sample

def get_datasets(
    dataset_root: str,
    split_seed: int = 42,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
    cache_rate: float = 0.05,
    num_workers: int = 4
):
    """
    Returns train, val, test datasets and dataloaders with appropriate transforms.
    Debug printout included.
    """
    train_dir = os.path.join(dataset_root, 'Training')
    print(f"DEBUG: train_dir={train_dir}")
    patient_to_cases = find_all_patients(train_dir)
    splits = patient_wise_split(patient_to_cases, val_pct, test_pct, seed=split_seed)

    train_entries = create_case_dicts(train_dir, splits['train'], has_label=True)
    val_entries = create_case_dicts(train_dir, splits['val'], has_label=True)
    test_entries = create_case_dicts(train_dir, splits['test'], has_label=True)

    print(f"DEBUG: Entries for splits: train={len(train_entries)}, val={len(val_entries)}, test={len(test_entries)}")

    # Get transforms
    train_transforms = get_train_transforms()
    val_transforms = get_test_val_transforms()

    train_ds = CacheDataset(train_entries, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
    val_ds = CacheDataset(val_entries, transform=val_transforms, cache_rate=cache_rate, num_workers=num_workers)
    test_ds = CacheDataset(test_entries, transform=val_transforms, cache_rate=cache_rate, num_workers=num_workers)

    return train_ds, val_ds, test_ds

def load_inference_cases(testing_dir: str) -> List[Dict]:
    """
    Loads cases from the /Testing folder for inference.
    Returns a list of dicts: { 'image': [modality_paths], 'case_id': str }
    Debug printout included.
    """
    pattern = os.path.join(testing_dir, '*')
    cases = [os.path.basename(d) for d in glob.glob(pattern) if os.path.isdir(d)]
    print(f"DEBUG: Found {len(cases)} inference case folders in {testing_dir}")
    print(f"DEBUG: Example inference cases: {cases[:10]}")
    data_list = []
    missing_cases = []
    for case in cases:
        case_dir = os.path.join(testing_dir, case)
        entry = {}
        try:
            entry['image'] = [os.path.join(case_dir, f"{case}{CASE_SUFFIX[m]}") for m in MODALITIES]
            entry['case_id'] = case
            all_exist = all([os.path.exists(p) for p in entry['image']])
            if not all_exist:
                missing_cases.append(case)
                logger.warning(f"Missing files for case {case}, skipping.")
                continue
            data_list.append(entry)
        except Exception as e:
            logger.error(f"Error processing inference case {case}: {e}")
    print(f"DEBUG: Valid inference cases loaded: {len(data_list)} (out of {len(cases)})")
    if missing_cases:
        print(f"DEBUG: Inference cases skipped due to missing files: {missing_cases[:5]} ... total {len(missing_cases)}")
    return data_list

def get_inference_dataset(
    testing_dir: str,
    cache_rate: float = 1.0,
    num_workers: int = 4
):
    """
    Returns a MONAI CacheDataset for inference-only /Testing set.
    """
    test_entries = load_inference_cases(testing_dir)
    test_transforms = get_inference_transforms()
    test_ds = CacheDataset(test_entries, transform=test_transforms, cache_rate=cache_rate, num_workers=num_workers)
    return test_ds

# For CLI or script usage
if __name__ == "__main__":
    import argparse
    data_dir = "../../Dataset/Segmentation/BraTS_GLI_PRE"
    parser = argparse.ArgumentParser(description="Prepare BraTS 2025 Datasets")
    parser.add_argument("--dataset_root", type=str, default=data_dir, help="Path to Dataset/Segmentation/BraTS_GLI_PRE/")
    parser.add_argument("--cache_rate", type=float, default=0.05, help="Cache rate for CacheDataset")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--inference", action="store_true", help="Load /Testing set for inference")
    args = parser.parse_args()

    if args.inference:
        testing_dir = os.path.join(args.dataset_root, 'Testing')
        test_ds = get_inference_dataset(testing_dir, cache_rate=args.cache_rate, num_workers=args.num_workers)
        print(f"Inference/Test set loaded: {len(test_ds)} cases.")
    else:
        train_ds, val_ds, test_ds = get_datasets(args.dataset_root, cache_rate=args.cache_rate, num_workers=args.num_workers)
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")