import gc
import random
import time
from itertools import islice

import numpy as np
import torch
from torch.utils.data import DataLoader

from criterions import load_criterion
from models import load_model
from optimizers import load_optimizer
import training as tr


class SizedDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class LimitedLoader:
    def __init__(self, loader, steps):
        self.steps = min(steps, len(loader))
        self.batch_size = loader.batch_size
        self.dataset = SizedDataset(
            min(len(loader.dataset), self.steps * loader.batch_size)
        )
        self.iterator = iter(loader) if self.steps > 0 else iter(())

    def __len__(self):
        return self.steps

    def __iter__(self):
        return islice(self.iterator, self.steps)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_loader(dataset, batch_size, shuffle, workers):
    # Worker와 prefetch 초기화 비용이 후보별 측정값에 반복해서 섞이지 않도록
    # warm-up 이후에도 worker를 유지한다.
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def make_phase_callback(callback, event_name):
    if callback is None:
        return None

    def phase_callback(event, batch, total_batches, **data):
        callback(
            event_name,
            batch=batch,
            total_batches=total_batches,
        )

    return phase_callback


def warm_up(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    steps,
    cutmix_prob,
    cutmix_alpha,
    callback=None,
):
    if steps == 0:
        return

    loader = LimitedLoader(train_loader, steps)

    tr.train(
        model,
        loader,
        optimizer,
        criterion,
        device=device,
        cutmix_prob=cutmix_prob,
        cutmix_alpha=cutmix_alpha,
        callback=make_phase_callback(callback, "warmup_batch_end"),
    )

    synchronize(device)


def warm_up_validation(model, test_loader, criterion, device, callback=None):
    loader = LimitedLoader(test_loader, 1)

    tr.test(
        model,
        loader,
        criterion,
        device=device,
        callback=make_phase_callback(
            callback,
            "validation_warmup_batch_end",
        ),
    )

    synchronize(device)


def validate_config(config, settings):
    batch_sizes = settings["batch_sizes"]

    if not batch_sizes or any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("batch_sizes는 1 이상의 정수 목록이어야 합니다.")
    if settings["workers"] < 0:
        raise ValueError("workers는 0 이상이어야 합니다.")
    if settings["warmup_steps"] < 0:
        raise ValueError("warmup_steps는 0 이상이어야 합니다.")
    if settings["steps"] <= 0:
        raise ValueError("steps는 1 이상이어야 합니다.")
    if not 0.0 <= config["cutmix"]["prob"] <= 1.0:
        raise ValueError("cutmix prob은 0과 1 사이여야 합니다.")


def benchmark_one(
    config,
    settings,
    batch_size,
    train_dataset,
    test_dataset,
    info,
    callback=None,
):
    device = config["model"]["device"]
    cutmix_prob = config["cutmix"]["prob"]
    cutmix_alpha = config["cutmix"]["alpha"]

    set_seed(config["training"]["seed"])

    train_loader = make_loader(
        train_dataset,
        batch_size,
        True,
        settings["workers"],
    )
    test_loader = make_loader(
        test_dataset,
        batch_size,
        False,
        settings["workers"],
    )

    warmup_batches = min(settings["warmup_steps"], len(train_loader))
    train_batches = min(settings["steps"], len(train_loader))
    test_batches = min(settings["steps"], len(test_loader))
    validation_warmup_batches = 0 if settings["train_only"] else 1

    if callback is not None:
        callback(
            "batch_size_start",
            batch_size=batch_size,
            warmup_batches=warmup_batches,
            validation_warmup_batches=validation_warmup_batches,
            train_batches=train_batches,
            test_batches=0 if settings["train_only"] else test_batches,
        )

    model = load_model(config["model"], info)
    if model is None:
        raise ValueError(
            f'{config["dataset"]["dataset"]}에서는 '
            f'{config["model"]["model"]}을 사용할 수 없습니다.'
        )

    optimizer = load_optimizer(config["optimizer"], model)
    criterion = load_criterion(config["criterion"], info).to(device)

    warm_up(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        settings["warmup_steps"],
        cutmix_prob,
        cutmix_alpha,
        callback,
    )

    if not settings["train_only"]:
        warm_up_validation(
            model,
            test_loader,
            criterion,
            device,
            callback,
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # iterator 생성은 측정 전에 끝낸다. 이 비용을 일부 batch의 시간에 넣고
    # 전체 epoch으로 확대하면 worker 시작 비용까지 여러 번 곱해지기 때문이다.
    measured_train_loader = LimitedLoader(train_loader, settings["steps"])

    synchronize(device)
    start = time.perf_counter()

    tr.train(
        model,
        measured_train_loader,
        optimizer,
        criterion,
        device=device,
        cutmix_prob=cutmix_prob,
        cutmix_alpha=cutmix_alpha,
        callback=callback,
    )

    synchronize(device)
    train_seconds = time.perf_counter() - start
    estimated_epoch_seconds = (
        train_seconds * len(train_loader) / len(measured_train_loader)
    )

    if not settings["train_only"]:
        measured_test_loader = LimitedLoader(test_loader, settings["steps"])

        synchronize(device)
        start = time.perf_counter()

        tr.test(
            model,
            measured_test_loader,
            criterion,
            device=device,
            callback=callback,
        )

        synchronize(device)
        test_seconds = time.perf_counter() - start
        estimated_epoch_seconds += (
            test_seconds * len(test_loader) / len(measured_test_loader)
        )

    peak_gib = None
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)

    return (
        train_seconds,
        estimated_epoch_seconds,
        len(measured_train_loader.dataset) / train_seconds,
        peak_gib,
    )


def release_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark(
    config,
    settings,
    train_dataset,
    test_dataset,
    info,
    callback=None,
):
    validate_config(config, settings)

    batch_sizes = settings["batch_sizes"]
    name = (
        f'{config["dataset"]["dataset"]}+'
        f'{config["model"]["model"]}+'
        f'{config["optimizer"]["optimizer"]}+'
        f'{config["criterion"]["criterion"]}'
    )

    if callback is not None:
        callback(
            "benchmark_start",
            batch_sizes=batch_sizes,
            steps=settings["steps"],
            name=name,
        )

    results = []

    for batch_size in batch_sizes:
        try:
            result = benchmark_one(
                config,
                settings,
                batch_size,
                train_dataset,
                test_dataset,
                info,
                callback,
            )
        except torch.cuda.OutOfMemoryError:
            if callback is not None:
                callback(
                    "batch_size_end",
                    batch_size=batch_size,
                    out_of_memory=True,
                    measured_seconds=None,
                    estimated_seconds=None,
                    throughput=None,
                    peak_gib=None,
                )
        else:
            measured_seconds, estimated_seconds, throughput, peak_gib = result
            results.append((batch_size, estimated_seconds))

            if callback is not None:
                callback(
                    "batch_size_end",
                    batch_size=batch_size,
                    out_of_memory=False,
                    measured_seconds=measured_seconds,
                    estimated_seconds=estimated_seconds,
                    throughput=throughput,
                    peak_gib=peak_gib,
                )
        finally:
            release_memory()

    best_batch = None
    best_seconds = None

    if results:
        best_batch, best_seconds = min(results, key=lambda item: item[1])

    if callback is not None:
        callback(
            "benchmark_end",
            best_batch=best_batch,
            best_seconds=best_seconds,
        )

    return results
