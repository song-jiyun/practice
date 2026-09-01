import torch
import curses
import random
import sys
import termios
import numpy as np

from datasets import get_dataset_list, load_dataset
from models import get_model_list, load_model
from optimizers import get_optimizer_list, load_optimizer
from schedulers import get_scheduler_list, load_scheduler, get_scheduler_config
from criterions import get_criterion_list, load_criterion
import training as tr
import benchmark as bm

from tqdm import tqdm
import datetime

class CursesLine:
    def __init__(self, stdscr, row):
        self.stdscr = stdscr
        self.row = row

    def write(self, text):
        text = text.replace("\r", "").replace("\n", "")

        if not text:
            return

        height, width = self.stdscr.getmaxyx()

        if self.row >= height:
            return

        try:
            self.stdscr.move(self.row, 0)

            self.stdscr.addnstr(
                self.row,
                0,
                text,
                max(0, width - 1),
            )

            self.stdscr.refresh()

        except curses.error:
            pass

    def flush(self):
        pass
    
    def clear(self):
        try:
            self.stdscr.move(self.row, 0)
            self.stdscr.clrtoeol()
            self.stdscr.refresh()

        except curses.error:
            pass

class TrainingUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr

        height, width = stdscr.getmaxyx()

        self.overall_height = 4
        self.log_height = height - self.overall_height

        self.overall_win = curses.newwin(self.overall_height, width, 0, 0)
        self.log_win = curses.newwin(self.log_height, width, self.overall_height, 0)

        self.log_win.scrollok(True)
        self.log_win.idlok(True)

        self.overall_writer = CursesLine(self.overall_win, 2)
        self.epoch_writer = None

        self.overall_progress = None
        self.epoch_progress = None        

        self.epoch_row = None

        self.next_row = 0

    def __call__(self, event, **data):
        if event == "training_start":
            self.training_start(**data)

        elif event == "epoch_start":
            self.epoch_start(**data)

        elif event == "train_batch_end":
            self.train_batch_end(**data)

        elif event == "test_batch_end":
            self.test_batch_end(**data)

        elif event == "epoch_end":
            self.epoch_end(**data)

        elif event == "training_end":
            self.training_end()

    def training_start(self, start_epoch, end_epoch, train_batches, test_batches, name):
        self.overall_win.erase()
        self.log_win.erase()

        self.overall_win.addstr(0, 0, f"Training {name}", curses.A_BOLD)

        width = max(40, self.stdscr.getmaxyx()[1] - 1)

        self.overall_progress = tqdm(
            total=end_epoch,
            initial=start_epoch - 1,
            desc="Overall",
            file=self.overall_writer,
            leave=True,
            position=0,
            dynamic_ncols=False,
            ncols=width,
            ascii=False,
            mininterval=0.1,
        )

        self.next_row = 0

        self.overall_win.refresh()
        self.log_win.refresh()

    def ensure_log_space(self, lines=3):
        if self.next_row + lines <= self.log_height:
            return

        scroll_lines = 3

        self.log_win.scroll(scroll_lines)

        self.next_row = max(0, self.next_row - scroll_lines)

        self.log_win.refresh()

    def epoch_start(self, epoch, end_epoch, train_batches, test_batches):
        self.ensure_log_space(3)

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        width = max(40, self.log_win.getmaxyx()[1] - 1)

        self.epoch_row = self.next_row

        self.epoch_writer = CursesLine(self.log_win, self.epoch_row)

        self.epoch_progress = tqdm(
            total=train_batches + test_batches,
            desc=f"[{ts}] Epoch {epoch}/{end_epoch}",
            file=self.epoch_writer,
            leave=True,
            position=0,
            dynamic_ncols=False,
            ncols=width,
            ascii=False,
            mininterval=0.1,
        )

    def train_batch_end(self, batch, total_batches, loss, processed, total_samples):
        self.epoch_progress.set_postfix_str(f"Train | Loss {loss:.4f}", refresh=False)
        self.epoch_progress.update(1)

    def test_batch_end(self, batch, total_batches, accuracy, processed, total_samples):
        self.epoch_progress.set_postfix_str(f"Test | Acc {accuracy:.2f}", refresh=False)
        self.epoch_progress.update(1)

    def epoch_end(self, epoch, train_loss, test_loss, accuracy, best_loss, best_accuracy):
        if self.epoch_progress is not None:
            self.epoch_progress.close()
            self.epoch_progress = None

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        width = max(40, self.log_win.getmaxyx()[1] - 1)

        result = f"[{ts}] Train Loss {train_loss:.4f} | Test Loss {test_loss:.4f} | Accuracy {accuracy:.2f}%"

        try:
            self.log_win.addnstr(self.epoch_row + 1, 0, result, width - 1)
        except curses.error:
            pass

        self.overall_progress.set_postfix_str(f"Acc {accuracy:.2f}% | Loss {test_loss:.4f}", refresh=False)
        self.overall_progress.update(1)

        # progress bar 바로 다음 줄로 이동
        self.next_row = self.epoch_row + 3

        self.log_win.refresh()
        self.overall_win.refresh()

    def training_end(self):
        if self.epoch_progress is not None:
            self.epoch_progress.close()
            self.epoch_progress = None

        if self.overall_progress is not None:
            self.overall_progress.close()

        self.ensure_log_space(2)

        width = max(40, self.log_win.getmaxyx()[1] - 1)

        try:
            self.log_win.addstr(self.next_row, 0, "Training completed.", curses.A_BOLD)
        except curses.error:
            pass

        self.log_win.refresh()
        self.overall_win.refresh()


class BenchmarkUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr

        height, width = stdscr.getmaxyx()

        self.overall_height = 4
        self.log_height = height - self.overall_height

        self.overall_win = curses.newwin(self.overall_height, width, 0, 0)
        self.log_win = curses.newwin(
            self.log_height,
            width,
            self.overall_height,
            0,
        )

        self.log_win.scrollok(True)
        self.log_win.idlok(True)

        self.overall_writer = CursesLine(self.overall_win, 2)
        self.batch_writer = None

        self.overall_progress = None
        self.batch_progress = None

        self.batch_row = None
        self.next_row = 0

    def __call__(self, event, **data):
        if event == "benchmark_start":
            self.benchmark_start(**data)

        elif event == "batch_size_start":
            self.batch_size_start(**data)

        elif event == "warmup_batch_end":
            self.batch_step_end("Warm-up")

        elif event == "validation_warmup_batch_end":
            self.batch_step_end("Validation warm-up")

        elif event == "train_batch_end":
            self.train_batch_end(**data)

        elif event == "test_batch_end":
            self.test_batch_end(**data)

        elif event == "batch_size_end":
            self.batch_size_end(**data)

        elif event == "benchmark_end":
            self.benchmark_end(**data)

    def benchmark_start(self, batch_sizes, steps, name):
        self.overall_win.erase()
        self.log_win.erase()

        self.overall_win.addstr(
            0,
            0,
            f"Benchmark {name} (up to {steps} batches/phase)",
            curses.A_BOLD,
        )

        width = max(40, self.stdscr.getmaxyx()[1] - 1)

        self.overall_progress = tqdm(
            total=len(batch_sizes),
            desc="Overall",
            file=self.overall_writer,
            leave=True,
            position=0,
            dynamic_ncols=False,
            ncols=width,
            ascii=False,
            mininterval=0.1,
        )

        self.next_row = 0

        self.overall_win.refresh()
        self.log_win.refresh()

    def ensure_log_space(self, lines=3):
        if self.next_row + lines <= self.log_height:
            return

        scroll_lines = 3

        self.log_win.scroll(scroll_lines)
        self.next_row = max(0, self.next_row - scroll_lines)
        self.log_win.refresh()

    def batch_size_start(
        self,
        batch_size,
        warmup_batches,
        validation_warmup_batches,
        train_batches,
        test_batches,
    ):
        self.ensure_log_space(3)

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        width = max(40, self.log_win.getmaxyx()[1] - 1)
        total_batches = (
            warmup_batches
            + validation_warmup_batches
            + train_batches
            + test_batches
        )

        self.batch_row = self.next_row
        self.batch_writer = CursesLine(self.log_win, self.batch_row)
        self.batch_progress = tqdm(
            total=total_batches,
            desc=f"[{ts}] Batch size {batch_size}",
            file=self.batch_writer,
            leave=True,
            position=0,
            dynamic_ncols=False,
            ncols=width,
            ascii=False,
            mininterval=0.1,
        )

    def batch_step_end(self, phase):
        if self.batch_progress is None:
            return

        self.batch_progress.set_postfix_str(phase, refresh=False)
        self.batch_progress.update(1)

    def train_batch_end(self, batch, total_batches, loss, processed, total_samples):
        if self.batch_progress is None:
            return

        self.batch_progress.set_postfix_str(
            f"Train | Loss {loss:.4f}",
            refresh=False,
        )
        self.batch_progress.update(1)

    def test_batch_end(self, batch, total_batches, accuracy, processed, total_samples):
        if self.batch_progress is None:
            return

        self.batch_progress.set_postfix_str(
            f"Test | Acc {accuracy:.2f}",
            refresh=False,
        )
        self.batch_progress.update(1)

    def batch_size_end(
        self,
        batch_size,
        out_of_memory,
        measured_seconds,
        estimated_seconds,
        throughput,
        peak_gib,
    ):
        if self.batch_progress is not None:
            self.batch_progress.close()
            self.batch_progress = None

        if self.batch_row is None:
            self.ensure_log_space(3)
            self.batch_row = self.next_row

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if out_of_memory:
            result = f"[{ts}] Batch size {batch_size} | OOM"
            postfix = f"Batch {batch_size} | OOM"
        else:
            peak = f"{peak_gib:.2f} GiB" if peak_gib is not None else "n/a"
            result = (
                f"[{ts}] Measured train {measured_seconds:.2f}s | "
                f"Estimated epoch {estimated_seconds:.2f}s | "
                f"{throughput:.1f} samples/s | Peak VRAM {peak}"
            )
            postfix = f"Batch {batch_size} | {estimated_seconds:.2f}s/epoch"

        width = max(40, self.log_win.getmaxyx()[1] - 1)

        try:
            self.log_win.addnstr(
                self.batch_row + 1,
                0,
                result,
                width - 1,
            )
        except curses.error:
            pass

        self.overall_progress.set_postfix_str(postfix, refresh=False)
        self.overall_progress.update(1)

        self.next_row = self.batch_row + 3
        self.batch_row = None

        self.log_win.refresh()
        self.overall_win.refresh()

    def benchmark_end(self, best_batch, best_seconds):
        if self.batch_progress is not None:
            self.batch_progress.close()
            self.batch_progress = None

        if self.overall_progress is not None:
            self.overall_progress.close()

        self.ensure_log_space(3)

        if best_batch is None:
            result = "No benchmark completed."
        else:
            result = (
                f"Best batch size: {best_batch} "
                f"({best_seconds:.2f}s/epoch)"
            )

        width = max(40, self.log_win.getmaxyx()[1] - 1)

        try:
            self.log_win.addnstr(
                self.next_row,
                0,
                result,
                width - 1,
                curses.A_BOLD,
            )
            self.log_win.addnstr(
                self.next_row + 1,
                0,
                "Press any key to return.",
                width - 1,
            )
        except curses.error:
            pass

        self.log_win.refresh()
        self.overall_win.refresh()


config = {
    "dataset": {
        "dataset": get_dataset_list()[0],
        "batch_size": 64,
    },

    "model": {
        "model": get_model_list()[0],
        "pretrained": True,
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    },

    "optimizer": {
        "optimizer": get_optimizer_list()[0],
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
    },

    "scheduler": get_scheduler_config(get_scheduler_list()[0]),

    "criterion": {
        "criterion": get_criterion_list()[0],
    },

    "cutmix": {    
        "prob": 0.5,
        "alpha": 1.0,
    },

    "training": {
        "epoch": 300,
        "seed": 42,
    }
}

def draw_menu(stdscr, title, options, selected):
    stdscr.clear()
    stdscr.addstr(0, 0, title, curses.A_BOLD)

    height, width = stdscr.getmaxyx()

    left_width = min(32, width // 2)
    right_x = left_width + 6

    for idx, opt in enumerate(options):
        y = 2 + idx * 2
        if idx == selected:
            stdscr.addstr(y, 2, f"> {opt}", curses.A_REVERSE)
        else:
            stdscr.addstr(y, 2, f"  {opt}")

    row = 0
    stdscr.addstr(row, right_x, "현재 설정", curses.A_BOLD)
    
    for key, value in config.items():
        row += 2
        if row < height - 1:
            stdscr.addstr(row, right_x, f"{key}")
            for key1, value1 in value.items():
                row += 1
                stdscr.addstr(row, right_x + 4, f"{key1}: {value1}")
            
    stdscr.refresh()

def input_float(stdscr, menu, selected):
    while True:
        draw_menu(stdscr, "", menu, selected)
        y = 2 + selected * 2
        x = len(menu[selected]) + 2 + 4
        stdscr.addstr(y, x, ": ", curses.A_REVERSE)
        curses.echo()
        value = stdscr.getstr(y, x + 2).decode()
        curses.noecho()

        try:
            return float(value)
        except ValueError:
            stdscr.addstr(y, x + 3, "숫자를 입력해주세요.")
            stdscr.getch()

def input_text(stdscr, prompt, default=""):
    stdscr.clear()
    stdscr.addstr(0, 0, prompt, curses.A_BOLD)
    if default:
        stdscr.addstr(2, 0, f"기본값: {default}")
    stdscr.addstr(4, 0, "> ")
    curses.curs_set(1)
    curses.echo()
    try:
        value = stdscr.getstr(4, 2).decode().strip()
    finally:
        curses.noecho()
        curses.curs_set(0)
    return value or default


def parse_batch_sizes(value):
    try:
        batch_sizes = sorted({
            int(item.strip())
            for item in value.split(",")
        })
    except ValueError as exc:
        raise ValueError(
            "batch size는 쉼표로 구분한 정수여야 합니다. 예: 64,96,128"
        ) from exc

    if not batch_sizes or batch_sizes[0] <= 0:
        raise ValueError("batch size는 1 이상의 정수여야 합니다.")

    return batch_sizes


def main(stdscr):
    curses.curs_set(0)
    if curses.has_colors():
        curses.start_color()
        try:
            curses.use_default_colors()
        except Exception:
            pass
        # pair 1: green text on default background
        curses.init_pair(1, -1, curses.COLOR_GREEN)
    
    stdscr.nodelay(False)
    selected = 0
    while True:
        menu = [
            "학습 설정",
            "학습 시작",
            "batch size 벤치마크",
            "나가기",
        ]
        draw_menu(stdscr, "머신러닝 실습", menu, selected)
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected == 0:
                #학습 설정
                configure(stdscr)
            elif selected == 1:
                #학습 시작
                start(stdscr)
                stdscr.getch()
            elif selected == 2:
                # 벤치마크 시작
                benchmark_menu(stdscr)
            elif selected == 3:
                #나가기
                return

def configure(stdscr):
    selected = 0
    while True:
        menu = [
            f"{key} 설정" for key, value in config.items()
        ]
        menu.append("뒤로가기")
        draw_menu(stdscr, "학습 설정", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected == 0:
                #dataset
                configure_dataset(stdscr)
            elif selected == 1:
                #model
                configure_model(stdscr)
            elif selected == 2:
                #optimizer
                configure_optimizer(stdscr)
            elif selected == 3:
                #scheduler
                configure_scheduler(stdscr)
            elif selected == 4:
                #criterion
                configure_criterion(stdscr)
            elif selected == 5:
                #cutmix
                configure_cutmix(stdscr)
            elif selected == 6:
                #training
                configure_training(stdscr)
            else:
                # 뒤로가기
                break

def configure_dataset(stdscr):    
    selected = 0
    while True:
        menu = [
            f"{key} 변경" for key, value in config['dataset'].items()
        ]
        menu.append("뒤로가기")

        draw_menu(stdscr, "dataset 설정", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected == 0:
                #dataset 변경
                change_dataset(stdscr)
            elif selected in range(1, len(menu) - 1):
                config['dataset'][list(config['dataset'].keys())[selected]] = int(input_float(stdscr, menu, selected))
            else:
                #뒤로가기
                break

def change_dataset(stdscr):
    selected = 0
    while True:
        menu = get_dataset_list()
        menu.append("뒤로가기")
        draw_menu(stdscr, "dataset 변경", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected in range(0, len(menu) - 1):
                config['dataset']['dataset'] = menu[selected]
            else:
                break

def configure_model(stdscr):
    selected = 0
    while True:
        menu = [
            f"{key} 변경" for key, value in config['model'].items()
        ]
        menu.append("뒤로가기")
        draw_menu(stdscr, "model 설정", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected == 0:
                #model 변경
                change_model(stdscr)
            elif selected == 1:
                #pretrained
                if config["model"]["pretrained"] == True:
                    config["model"]["pretrained"] = False
                else:
                    config["model"]["pretrained"] = True
            elif selected == 2:
                #device
                if config["model"]["device"] == torch.device("cuda"):
                    config["model"]["device"] = torch.device("cpu")
                else:
                    config["model"]["device"] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            else:
                #뒤로가기
                break

def change_model(stdscr):
    selected = 0
    while True:
        menu = get_model_list()
        menu.append("뒤로가기")
        draw_menu(stdscr, "model 변경", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected in range(0, len(menu) - 1):
                config['model']['model'] = menu[selected]
            else:
                break

def configure_optimizer(stdscr):
    selected = 0
    while True:
        menu = [
            f"{key} 변경" for key, value in config['optimizer'].items()
        ]
        menu.append("뒤로가기")
        draw_menu(stdscr, "optimizer 설정", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected == 0:
                #optimizer 변경
                change_optimizer(stdscr)
            elif selected in range(1, len(menu) - 1):
                config['optimizer'][list(config['optimizer'].keys())[selected]] = input_float(stdscr, menu, selected)
            else:
                #뒤로가기
                break

def change_optimizer(stdscr):
    selected = 0
    while True:
        menu = get_optimizer_list()
        menu.append("뒤로가기")
        draw_menu(stdscr, "optimizer 변경", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected in range(0, len(menu) - 1):
                config['optimizer']['optimizer'] = menu[selected]
            else:
                break

def configure_scheduler(stdscr):
    selected = 0
    while True:
        menu = [
            f"{key} 변경" for key, value in config['scheduler'].items()
        ]
        menu.append("뒤로가기")
        draw_menu(stdscr, "scheduler 설정", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected == 0:
                #scheduler 변경
                change_scheduler(stdscr)
            elif selected in range(1, len(menu) - 1):
                config['scheduler'][list(config['scheduler'].keys())[selected]] = input_float(stdscr, menu, selected)
            else:
                #뒤로가기
                break

def change_scheduler(stdscr):
    selected = 0
    while True:
        menu = get_scheduler_list()
        menu.append("뒤로가기")
        draw_menu(stdscr, "scheduler 변경", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected in range(0, len(menu) - 1):
                config['scheduler'] = get_scheduler_config(menu[selected])
            else:
                break

def configure_criterion(stdscr):
    selected = 0
    while True:
        menu = [
            f"{key} 변경" for key, value in config['criterion'].items()
        ]
        menu.append("뒤로가기")
        draw_menu(stdscr, "criterion 설정", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected == 0:
                #criterion 변경
                change_criterion(stdscr)
            elif selected == 1:
                #뒤로가기
                break

def change_criterion(stdscr):
    selected = 0
    while True:
        menu = get_criterion_list()
        menu.append("뒤로가기")
        draw_menu(stdscr, "criterion 변경", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected in range(0, len(menu) - 1):
                config['criterion']['criterion'] = menu[selected]
            else:
                break

def configure_cutmix(stdscr):
    selected = 0
    while True:
        menu = [
            f"{key} 변경" for key, value in config['cutmix'].items()
        ]
        menu.append("뒤로가기")
        draw_menu(stdscr, "cutmix 설정", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected in range(0, len(menu) - 1):
                #cutmix 변경
                config['cutmix'][list(config['cutmix'].keys())[selected]] = input_float(stdscr, menu, selected)
            else:
                #뒤로가기
                break

def configure_training(stdscr):
    selected = 0
    while True:
        menu = [
            f"{key} 변경" for key, value in config['training'].items()
        ]
        menu.append("뒤로가기")
        draw_menu(stdscr, "training 설정", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected in range(0, len(menu) - 1):
                #training 변경
                config['training'][list(config['training'].keys())[selected]] = int(input_float(stdscr, menu, selected))
            else:
                #뒤로가기
                break

def get_default_batch_sizes(batch_size):
    return sorted({
        max(1, batch_size // 2),
        max(1, batch_size * 3 // 4),
        batch_size,
        batch_size * 5 // 4,
        batch_size * 3 // 2,
        batch_size * 2,
    })


def draw_benchmark_menu(stdscr, settings, selected):
    batch_sizes = ",".join(str(value) for value in settings["batch_sizes"])
    menu = [
        f"batch_sizes: {batch_sizes}",
        f"workers: {settings['workers']}",
        f"warmup_steps: {settings['warmup_steps']}",
        f"steps: {settings['steps']}",
        f"train_only: {settings['train_only']}",
        "벤치마크 실행",
        "뒤로가기",
    ]

    stdscr.clear()
    stdscr.addstr(0, 0, "Batch size 벤치마크 설정", curses.A_BOLD)

    height, width = stdscr.getmaxyx()
    context_x = min(42, width // 2 + 8)
    menu_width = context_x - 4 if context_x < width - 1 else width - 3

    for idx, option in enumerate(menu):
        y = 2 + idx * 2
        if y >= height:
            break

        attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL
        label = f"> {option}" if idx == selected else f"  {option}"
        stdscr.addnstr(
            y,
            2,
            label,
            max(0, menu_width),
            attr,
        )

    if context_x < width - 1:
        stdscr.addstr(0, context_x, "현재 학습 구성", curses.A_BOLD)
        context = [
            f"dataset: {config['dataset']['dataset']}",
            f"model: {config['model']['model']}",
            f"optimizer: {config['optimizer']['optimizer']}",
            f"criterion: {config['criterion']['criterion']}",
            f"device: {config['model']['device']}",
        ]

        for idx, value in enumerate(context, start=1):
            row = idx + 1
            if row >= height:
                break
            stdscr.addnstr(
                row,
                context_x,
                value,
                max(0, width - context_x - 1),
            )

    stdscr.refresh()


def show_input_error(stdscr, message):
    stdscr.clear()
    stdscr.addstr(0, 0, "입력 오류", curses.A_BOLD)
    stdscr.addstr(2, 0, message)
    stdscr.addstr(4, 0, "아무 키나 누르면 돌아갑니다.")
    stdscr.refresh()
    stdscr.getch()


def benchmark_menu(stdscr):
    settings = {
        "batch_sizes": get_default_batch_sizes(
            config["dataset"]["batch_size"]
        ),
        "workers": 4,
        "warmup_steps": 10,
        "steps": 100,
        "train_only": False,
    }
    selected = 0
    menu_size = 7

    while True:
        draw_benchmark_menu(stdscr, settings, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % menu_size
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % menu_size
        elif key in (10, 13, ord(" ")):
            if selected == 0:
                default = ",".join(
                    str(value)
                    for value in settings["batch_sizes"]
                )
                raw_value = input_text(
                    stdscr,
                    "측정할 batch size를 쉼표로 구분해 입력하세요.",
                    default,
                )

                try:
                    settings["batch_sizes"] = parse_batch_sizes(raw_value)
                except ValueError as exc:
                    show_input_error(stdscr, str(exc))

            elif selected in (1, 2, 3):
                setting = ("workers", "warmup_steps", "steps")[selected - 1]
                raw_value = input_text(
                    stdscr,
                    f"{setting} 값을 입력하세요.",
                    str(settings[setting]),
                )

                try:
                    value = int(raw_value)
                    if setting in ("workers", "warmup_steps") and value < 0:
                        raise ValueError(f"{setting}는 0 이상이어야 합니다.")
                    if setting == "steps" and value <= 0:
                        raise ValueError("steps는 1 이상이어야 합니다.")
                except ValueError as exc:
                    message = str(exc) or "정수를 입력해주세요."
                    show_input_error(stdscr, message)
                else:
                    settings[setting] = value

            elif selected == 4:
                settings["train_only"] = not settings["train_only"]

            elif selected == 5:
                start_benchmark(stdscr, settings)
                stdscr.getch()

            else:
                return

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(True)


def show_runtime_error(stdscr, exc):
    height, width = stdscr.getmaxyx()
    message = f"오류가 발생했습니다: {exc}"

    try:
        stdscr.addnstr(
            min(4, height - 1),
            4,
            message,
            max(0, width - 5),
            curses.A_BOLD,
        )
        stdscr.refresh()
    except curses.error:
        pass


def start(stdscr):
    try:
        set_seed(config["training"]["seed"])

        train_loader, test_loader, info = load_dataset(config["dataset"])
        
        model = load_model(config["model"], info)
        if model is None:
            msg = f'{config["dataset"]["dataset"]}에서는 {config["model"]["model"]}을 사용할 수 없습니다.'
            attr = curses.color_pair(1) | curses.A_BOLD if curses.has_colors() else curses.A_BOLD
            stdscr.addstr(4, 4, msg, curses.A_REVERSE)
            return

        optimizer = load_optimizer(config["optimizer"], model)
        scheduler = load_scheduler(config["scheduler"], optimizer, config["training"]["epoch"])
        criterion = load_criterion(config["criterion"], info)

        ui = TrainingUI(stdscr)
      
        tr.training(config, model, train_loader, test_loader, optimizer, scheduler, criterion, callback=ui)
    except KeyboardInterrupt:
        return
    except Exception as exc:
        show_runtime_error(stdscr, exc)
        return


def start_benchmark(stdscr, settings):
    try:
        set_seed(config["training"]["seed"])
        bm.validate_config(config, settings)

        dataset_config = {
            "dataset": config["dataset"]["dataset"],
            "batch_size": settings["batch_sizes"][0],
        }
        train_loader, test_loader, info = load_dataset(dataset_config)
        train_dataset = train_loader.dataset
        test_dataset = test_loader.dataset
        del train_loader, test_loader

        ui = BenchmarkUI(stdscr)

        bm.benchmark(
            config,
            settings,
            train_dataset,
            test_dataset,
            info,
            callback=ui,
        )
    except KeyboardInterrupt:
        return
    except Exception as exc:
        show_runtime_error(stdscr, exc)
        return


def run():
    terminal_state = None

    if sys.stdin.isatty():
        try:
            terminal_state = termios.tcgetattr(sys.stdin.fileno())
        except termios.error:
            pass

    try:
        curses.wrapper(main)
    finally:
        if terminal_state is not None:
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    terminal_state,
                )
            except (termios.error, OSError):
                pass

if __name__ == "__main__":
    run()
