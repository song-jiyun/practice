import curses
from dataset import get_dataset_list
from model import get_model_list
from optimizer import get_optimizer_list
from scheduler import get_scheduler_list
from criterion import get_criterion_list

config = {
    "dataset": {
        "dataset": get_dataset_list()[0],
    },

    "model": {
        "model": get_model_list()[0],
    },

    "optimizer": {
        "optimizer": get_optimizer_list()[0],
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
    },

    "scheduler": {
        "scheduler": get_scheduler_list()[0],
    },

    "criterion": {
        "criterion": get_criterion_list()[0],
    },

    "training": {
        "epochs": 300,
        "batch_size": 64,
    },

    "cutmix": {    
        "prob": 0.5,
        "alpha": 1.0,
    },
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

def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(False)

    menu = [
        "학습 설정",
        "학습 시작",
        "나가기",
    ]
    selected = 0

    while True:
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
            elif selected == 2:
                #나가기
                return

def configure(stdscr):
    menu = [
        f"{key} 설정" for key, value in config.items()
    ]
    menu.append("뒤로가기")
    selected = 0
    while True:
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
                #training
                configure_training(stdscr)
            elif selected == 6:
                #cutmix
                configure_cutmix(stdscr)
            elif selected == 7:
                # 뒤로가기
                break

def configure_dataset(stdscr):
    menu = [
        f"{key} 변경" for key, value in config['dataset'].items()
    ]
    menu.append("뒤로가기")
    selected = 0
    while True:
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
            elif selected == 1:
                #뒤로가기
                break

def change_dataset(stdscr):
    menu = get_dataset_list()
    menu.append("뒤로가기")
    selected = 0
    while True:
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
    menu = [
        f"{key} 변경" for key, value in config['model'].items()
    ]
    menu.append("뒤로가기")
    selected = 0
    while True:
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
                #뒤로가기
                break

def change_model(stdscr):
    menu = get_model_list()
    menu.append("뒤로가기")
    selected = 0
    while True:
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
    menu = [
        f"{key} 변경" for key, value in config['optimizer'].items()
    ]
    menu.append("뒤로가기")
    selected = 0
    while True:
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
    menu = get_optimizer_list()
    menu.append("뒤로가기")
    selected = 0
    while True:
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
    menu = [
        f"{key} 변경" for key, value in config['scheduler'].items()
    ]
    menu.append("뒤로가기")
    selected = 0
    while True:
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
            elif selected == 1:
                #뒤로가기
                break

def change_scheduler(stdscr):
    menu = get_scheduler_list()
    menu.append("뒤로가기")
    selected = 0
    while True:
        draw_menu(stdscr, "scheduler 변경", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected in range(0, len(menu) - 1):
                config['scheduler']['scheduler'] = menu[selected]
            else:
                break

def configure_criterion(stdscr):
    menu = [
        f"{key} 변경" for key, value in config['criterion'].items()
    ]
    menu.append("뒤로가기")
    selected = 0
    while True:
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
    menu = get_criterion_list()
    menu.append("뒤로가기")
    selected = 0
    while True:
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

def configure_training(stdscr):
    menu = [
        f"{key} 변경" for key, value in config['training'].items()
    ]
    menu.append("뒤로가기")
    selected = 0
    while True:
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

def configure_cutmix(stdscr):
    menu = [
        f"{key} 변경" for key, value in config['cutmix'].items()
    ]
    menu.append("뒤로가기")
    selected = 0
    while True:
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

def start(stdscr):
    pass

if __name__ == "__main__":
    curses.wrapper(main)