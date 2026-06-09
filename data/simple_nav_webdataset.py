

import argparse
import base64
import getpass
import io
import json
import os
import os.path as osp
import pickle
import pprint
import shutil
import tarfile
from bisect import bisect
from functools import lru_cache, reduce
from multiprocessing.pool import ThreadPool as Pool

import torch
import torch.distributed
from filelock import FileLock, Timeout
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, get_worker_info

from qwen3_vl.wids import ShardListDataset


def load_tarfile(tar_path):
    return tarfile.open(tar_path)


DEFAULT_WEBDS_DATA_PATH = "~/datasets/captioning/webds"


def save_json(obj, fpath):
    print(f"saving to {fpath}")
    os.makedirs(osp.dirname(fpath), exist_ok=True)
    json.dump(obj, open(fpath, "w+"), indent=2)


def generate_and_load_tar_meta(data_path, tar_path, cache_dir, overwrite=False):
    tar_abspath = osp.abspath(osp.join(data_path, tar_path))
    tar_abs_metapath = osp.join(
        osp.expanduser(cache_dir),
        "dev",
        tar_abspath.replace("/", "--") + ".wdsmeta.json",
    )
    tar_real_metapath = osp.join(
        osp.expanduser(cache_dir),
        "dev",
        osp.realpath(tar_abspath).replace("/", "--") + ".wdsmeta.json",
    )

    if not osp.exists(tar_abs_metapath) and not osp.exists(tar_real_metapath) or overwrite:
        # generate meta information for both abs and real file paths
        print(f"    Generating meta: {tar_abs_metapath}")
        try:
            tar = load_tarfile(tar_abspath)
            uuids = list({osp.splitext(_)[0] for _ in tar.getnames()})
        except tarfile.ReadError as e:
            print(f"Skipping {tar_abspath}")
            print(e)
            return None
        nsamples = len(uuids)

        tar_meta = {
            "url": osp.abspath(tar_abspath),
            "nsamples": nsamples,
            "filesize": osp.getsize(tar_abspath),
        }
        save_json(tar_meta, tar_abs_metapath)

        tar_meta = {
            "url": osp.realpath(tar_abspath),
            "nsamples": nsamples,
            "filesize": osp.getsize(tar_abspath),
        }
        save_json(tar_meta, tar_real_metapath)

    if osp.exists(tar_abs_metapath):
        print(f"    Loading abs meta: {tar_abs_metapath}")
        tar_meta = json.load(open(tar_abs_metapath))
    elif osp.exists(tar_real_metapath):
        print(f"    Loading realpath meta: {tar_real_metapath}")
        tar_meta = json.load(open(tar_real_metapath))
    else:
        return None
    return tar_meta


def generate_wids_meta(tar_list, data_path, cache_dir, idx=0, total=0):
    meta_path_of_tar_abs = osp.join(
        osp.expanduser(cache_dir),
        data_path.replace("/", "--") + ".wdsmeta.json",
    )
    meta_path_of_tar_rel = osp.join(osp.expanduser(data_path), "wids-meta.json")
    meta = {
        "name": "coyo-dev",
        "__kind__": "Nav-WebDataset",
        "wids_version": 1,
        "shardlist": [],
    }

    for idx, tar_path in enumerate(tar_list):
        print(f"{idx}-of-{len(tar_list)}")
        tar_meta = generate_and_load_tar_meta(data_path, tar_path, cache_dir)
        tar_meta["url"] = osp.abspath(osp.join(data_path, tar_path))
        meta["shardlist"].append(tar_meta)

    meta["shardlist"] = sorted(meta["shardlist"], key=lambda x: x["url"])
    if total == 0:
        save_json(meta, meta_path_of_tar_abs)

    meta = {
        "name": "coyo-dev",
        "__kind__": "Nav-WebDataset",
        "wids_version": 1,
        "shardlist": [],
    }
    for idx, tar_path in enumerate(tar_list):
        print(f"{idx}-of-{len(tar_list)}")
        tar_meta = generate_and_load_tar_meta(data_path, tar_path, cache_dir)
        if tar_meta is None:
            continue
        tar_meta["url"] = tar_path
        meta["shardlist"].append(tar_meta)

    meta["shardlist"] = sorted(meta["shardlist"], key=lambda x: x["url"])
    if total == 0:
        save_json(meta, meta_path_of_tar_rel)


def prepare_wids_meta(data_path, cache_dir="~/datasets/nav-webds-meta", idx=0, total=0):
    cache_dir = osp.expanduser(cache_dir)
    tar_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            fpath = osp.join(root, file)
            fpath = osp.relpath(fpath, data_path)
            if not fpath.endswith(".tar"):
                continue
            tar_list.append(fpath)
    tar_list = sorted(tar_list)

    if total > 0:
        chunk = len(tar_list) // total
        begin_idx = chunk * idx
        end_idx = chunk * (idx + 1)
        if idx == total - 1:
            end_idx = len(tar_list)
        tar_list = tar_list[begin_idx:end_idx]
        print(f"{chunk}, {begin_idx} -> {end_idx}")

    assert len(tar_list) > 0, f"no tar was found in the repository {data_path} !"
    print(f"generating meta for total {len(tar_list)} files.")
    generate_wids_meta(tar_list, data_path, cache_dir, idx=idx, total=total)


class NavWebDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path=DEFAULT_WEBDS_DATA_PATH,
        meta_path=None,
        cache_dir="~/datasets/nav-webds-meta",
        max_shards_to_load=None,
    ):
        self.data_path = osp.expanduser(data_path)
        self.meta_path = osp.expanduser(meta_path) if meta_path is not None else None

        _local_meta_path = osp.join(self.data_path, "wids-meta.json")
        if meta_path is None and osp.exists(_local_meta_path):
            print(f"loading from {_local_meta_path}")
            self.meta_path = meta_path = _local_meta_path

        if meta_path is None:
            self.meta_path = osp.join(
                osp.expanduser(cache_dir),
                self.data_path.replace("/", "--") + f".max_shards:{max_shards_to_load}" + ".wdsmeta.json",
            )

        assert osp.exists(self.meta_path), f"meta path not found in [{self.meta_path}] or [{_local_meta_path}]"
        print(f"[Nav-WebDataset] Loading meta information {self.meta_path}", flush=True)

        import hashlib

        uuid = hashlib.sha256(self.meta_path.encode()).hexdigest()[:8]
        self.dataset = ShardListDataset(
            self.meta_path,
            cache_dir=osp.expanduser(f"~/.cache/_wids_cache/{getpass.getuser()}-{uuid}"),
        )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def simple_collate(batch):
        batched_data = {}
        for data in batch:
            for k, v in data.items():
                if k not in batched_data:
                    batched_data[k] = []
                batched_data[k].append(v)
        return dict(batched_data)

    @staticmethod
    def custom_collate(batch):
        def transform2list(a: dict):
            for k, v in a.items():
                if isinstance(v, dict):
                    a[k] = transform2list(v)
                else:
                    a[k] = [v]
            return a

        def merge(a: dict, b: dict, path=[], strict=False):
            c = {}
            keys = set(a.keys()).union(b.keys())
            for key in keys:
                if key in a and key in b:
                    if isinstance(a[key], dict) and isinstance(b[key], dict):
                        c[key] = merge(a[key], b[key], path + [str(key)], strict=strict)
                    else:
                        c[key] = a[key] + b[key]
                else:
                    if strict:
                        raise Exception("Conflict at " + ".".join(path + [str(key)]))
                    c[key] = a[key] if key in a else b[key]
            return c

        tasks = (transform2list(_) for _ in batch)
        return reduce(merge, tasks)


if __name__ == "__main__":
    import torch
    from torch.utils.data.distributed import DistributedSampler

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", nargs="?", type=str)
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("--shards", type=int, default=0)
    parser.add_argument("--total", type=int, default=0)
    parser.add_argument("--test-all", action="store_true")
    args = parser.parse_args()

    print("Data path: ", args.data_path)
    prepare_wids_meta(args.data_path, idx=args.shards, total=args.total)

    if args.total > 0:
        print("building meta information only")
        exit(0)

    train_dataset = NavWebDataset(data_path=args.data_path)
    print("dataset size: ", len(train_dataset))
    print(train_dataset[0])

    if args.test_all:
        print("iterating all dataset for data integrity.")
        train_dataset = NavWebDataset(data_path=args.data_path)
        dloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=False,
            sampler=None,
            batch_size=8,
            collate_fn=NavWebDataset.custom_collate,
            num_workers=8,
        )
        print(len(train_dataset), len(dloader))
        for idx, data in enumerate(dloader):
            if ".json" in data and ".mp4" in data:
                print(f"{idx}-of-{len(dloader)}", type(data))
            if idx >= 5:
                break
