import curses
from dataset import get_dataset_list
from model import get_model_list
from optimizer import get_optimizer_list
from scheduler import get_scheduler_list
from criterion import get_criterion_list

config = {
    "dataset": get_dataset_list()[0],
    "model": get_model_list()[0],
    "optimizer": get_optimizer_list()[0],
    "scheduler": get_scheduler_list()[0],
    "criterion": get_criterion_list()[0],
    "epochs": 300,
    "batch_size": 64,
    "lr": 0.05,
    "cutmix_prob": 0.5,
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
            stdscr.addstr(row, right_x, f"{key}: {value}")
            

    stdscr.refresh()

def configure_dataset(stdscr):
    menu = get_dataset_list()
    menu.append("뒤로가기")
    selected = 0
    while True:
        draw_menu(stdscr, "데이터셋 변경", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected in range(0, len(menu) - 1):
                config['dataset'] = menu[selected]
            else:
                break

def configure_model(stdscr):
    menu = get_model_list()
    menu.append("뒤로가기")
    selected = 0
    while True:
        draw_menu(stdscr, "모델 변경", menu, selected)
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord('k')):
            selected = (selected - 1) % len(menu)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected = (selected + 1) % len(menu)
        elif key in (10, 13, ord(' ')):
            if selected in range(0, len(menu) - 1):
                config['model'] = menu[selected]
            else:
                break
    pass

def configure_hyperparameters(stdscr):
    pass

def configure(stdscr):
    menu = [
        "데이터셋 변경",
        "모델 변경",
        "하이퍼파라미터 변경",
        "뒤로가기"
    ]
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
                configure_dataset(stdscr)
            elif selected == 1:
                configure_model(stdscr)
            elif selected == 2:
                configure_hyperparameters(stdscr)
            elif selected == 3:
                # 뒤로가기
                break

def start(stdscr):
    pass

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
                configure(stdscr)
                pass
            elif selected == 1:
                start(stdscr)
                pass
            elif selected == 2:
                return

if __name__ == "__main__":
    curses.wrapper(main)