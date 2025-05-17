import mss
import cv2
import numpy as np
import pyautogui
import time

# --- НАСТРОЙКИ ---
GAME_REGION = {"top": 245, "left": 30 , "width": 850  , "height": 200} # Область окна игры
    
# --- ПЕРВАЯ (нижняя) ОБЛАСТЬ ОБНАРУЖЕНИЯ НАЗЕМНЫХ ПРЕПЯТСТВИЙ ---
# Начальная позиция левого края
INITIAL_LONG_RANGE_X = 152          
# Фиксированная ширина  
LONG_RANGE_WIDTH = 10  
   
LONG_RANGE_Y = 112    # Начальная координата Y
LONG_RANGE_HEIGHT = 40 # Высота
  
#  Параметры адаптации (сдвиг) для нижней области
LONG_RANGE_SHIFT_INCREMENT = 186.9  # Насколько сдвигать вправо при каждом интервале
# Максимальная позиция левого края
MAX_LONG_RANGE_X = 224

# --- ВТОРАЯ (верхней) ОБЛАСТЬ ОБНАРУЖЕНИЯ НАЗЕМНЫХ ПРЕПЯТСТВИЙ ---
# Начальная позиция левого края
INITIAL_SHORT_RANGE_X = 152
SHORT_RANGE_WIDTH = 12

SHORT_RANGE_Y = 102   # Y координата 
SHORT_RANGE_HEIGHT = 11 # Высота 

# Параметры адаптации (сдвиг) для верхней области
SHORT_RANGE_SHIFT_INCREMENT = 16.9  # Насколько сдвигать вправо при каждом интервале
# Максимальная позиция левого края
MAX_SHORT_RANGE_X = 224   
 
 
#  Параметры ОБЩЕЙ адаптации по времени
ADAPTATION_INTERVAL = 7.3  # Интервал времени между адаптациями
 
#  Параметры адаптации таймингов приземления
FAST_LAND_DELAY_DECREMENT_PER_INTERVAL = 0.0165 # Насколько уменьшать задержку быстрого приземления при каждом интервале адаптации
MIN_FAST_LAND_DELAY = 0.01 # Минимальная задержка быстрого приземления 

  
# Пороги для обнаружения препятствий для ПРЫЖКА 
# Проверяем пиксели на темный цвет
OBSTACLE_COLOR_THRESHOLD_JUMP = 100
OBSTACLE_PIXEL_COUNT_THRESHOLD = 0.9


FAST_LAND_DELAY_AFTER_JUMP = 0.148 # Начальная задержка перед зажатием "вниз" после прыжка.
HOLD_DOWN_DURATION = 0.03 # Длительность зажатия кнопки "вниз" для быстрого приземления
HOLD_DOWN_DOWN_DURATION = 0.08 # Задержка при наклоне

sct = mss.mss()

current_long_range_x = INITIAL_LONG_RANGE_X
current_short_range_x = INITIAL_SHORT_RANGE_X

last_adaptation_time = time.time()
first_jump_performed = False


def detect_obstacles(game_screenshot_bgr, long_range_x, short_range_x):
    obstacle_detected = False

    area1_left_x = int(long_range_x)
    area1_right_x = int(long_range_x + LONG_RANGE_WIDTH)


    area1_right_x = min(area1_right_x, game_screenshot_bgr.shape[1])
    area1_left_x = min(area1_left_x, game_screenshot_bgr.shape[1] - 1)

    if area1_right_x > area1_left_x and area1_left_x >= 0:
        area1_roi = game_screenshot_bgr[int(LONG_RANGE_Y) : int(LONG_RANGE_Y) + int(LONG_RANGE_HEIGHT),
                                        area1_left_x : area1_right_x]

        if area1_roi.size > 0:
            _, thresh_area1 = cv2.threshold(cv2.cvtColor(area1_roi, cv2.COLOR_BGR2GRAY), OBSTACLE_COLOR_THRESHOLD_JUMP, 255, cv2.THRESH_BINARY_INV)
            obstacle_pixels_1 = cv2.countNonZero(thresh_area1)

            if obstacle_pixels_1 > OBSTACLE_PIXEL_COUNT_THRESHOLD:
                obstacle_detected = 1

    
    area2_left_x = int(short_range_x)
    area2_right_x = int(short_range_x + SHORT_RANGE_WIDTH)

    area2_right_x = min(area2_right_x, game_screenshot_bgr.shape[1])
    area2_left_x = min(area2_left_x, game_screenshot_bgr.shape[1] - 1)

    if area2_right_x > area2_left_x and area2_left_x >= 0:
        area2_roi = game_screenshot_bgr[int(SHORT_RANGE_Y) : int(SHORT_RANGE_Y) + int(SHORT_RANGE_HEIGHT),
                                        area2_left_x : area2_right_x]

        if area2_roi.size > 0:
            _, thresh_area2 = cv2.threshold(cv2.cvtColor(area2_roi, cv2.COLOR_BGR2GRAY), OBSTACLE_COLOR_THRESHOLD_JUMP, 255, cv2.THRESH_BINARY_INV)
            obstacle_pixels_2 = cv2.countNonZero(thresh_area2)

            if obstacle_pixels_2 > OBSTACLE_PIXEL_COUNT_THRESHOLD:
                obstacle_detected = 2


    if obstacle_detected == 1:
        return "jump"
    if obstacle_detected == 2:
        return "down"
    else:
        return "none"


def perform_action(action):
    global first_jump_performed, last_adaptation_time 
    global FAST_LAND_DELAY_AFTER_JUMP 
    global HOLD_DOWN_DOWN_DURATION

    if action == "jump":
        pyautogui.press('space')

        if not first_jump_performed:
             first_jump_performed = True
             last_adaptation_time = time.time()
             print("Первый прыжок выполнен. Адаптация по времени (сдвиг и тайминги) активирована.")


        time.sleep(FAST_LAND_DELAY_AFTER_JUMP)
        pyautogui.keyDown('down')
        time.sleep(HOLD_DOWN_DURATION)
        pyautogui.keyUp('down')
    if action == "down":
        pyautogui.keyDown('down')
        time.sleep(HOLD_DOWN_DOWN_DURATION)
        pyautogui.keyUp('down')


def is_game_over(game_screenshot_bgr):
    return False


print("Приготовьтесь...")
time.sleep(3) 
print("Запуск бота!")
pyautogui.press('space')


try:
    while True:
        current_time = time.time()
        if first_jump_performed and (current_time - last_adaptation_time) > ADAPTATION_INTERVAL:
            current_long_range_x = min(current_long_range_x + LONG_RANGE_SHIFT_INCREMENT, MAX_LONG_RANGE_X)
            print(f"Время: {current_time:.2f}. область сдвинута. Новый левый край X: {current_long_range_x}")

            current_short_range_x = min(current_short_range_x + SHORT_RANGE_SHIFT_INCREMENT, MAX_SHORT_RANGE_X)
            print(f"Время: {current_time:.2f}. верхняя область сдвинута. Новый левый край X: {current_short_range_x}")

            HOLD_DOWN_DOWN_DURATION += 0.09
            FAST_LAND_DELAY_AFTER_JUMP = max(FAST_LAND_DELAY_AFTER_JUMP - FAST_LAND_DELAY_DECREMENT_PER_INTERVAL, MIN_FAST_LAND_DELAY)
            print(f"Задержка быстрого приземления уменьшена до: {FAST_LAND_DELAY_AFTER_JUMP:.3f}")

            last_adaptation_time = current_time


        sct = mss.mss()
        sct_img = sct.grab(GAME_REGION)
        game_screenshot_raw = np.array(sct_img)
        game_screenshot_bgr = cv2.cvtColor(game_screenshot_raw, cv2.COLOR_BGRA2BGR)

        if game_screenshot_bgr is None or game_screenshot_bgr.size == 0:
             time.sleep(0.01) 
             continue

        action = detect_obstacles(game_screenshot_bgr, current_long_range_x, current_short_range_x)

        if action == "jump":
            perform_action(action)
        elif action == "down":
            perform_action(action)

        game_screenshot_display = game_screenshot_bgr.copy()

        cv2.rectangle(game_screenshot_display,
                      (int(current_long_range_x), int(LONG_RANGE_Y)),
                      (int(current_long_range_x + LONG_RANGE_WIDTH), int(LONG_RANGE_Y + LONG_RANGE_HEIGHT)),
                      (0, 255, 0), 1) 
        cv2.rectangle(game_screenshot_display,
                      (int(current_short_range_x), int(SHORT_RANGE_Y)),
                      (int(current_short_range_x + SHORT_RANGE_WIDTH), int(SHORT_RANGE_Y + SHORT_RANGE_HEIGHT)),
                      (255, 0, 0), 1)



        cv2.imshow("Игровая область с двойными зонами обнаружения", game_screenshot_display)


        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Бот остановлен вручную.")

finally:
    cv2.destroyAllWindows()
