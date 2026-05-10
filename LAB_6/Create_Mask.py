import cv2
import numpy as np


def create_morphology_mask(img):
    """
    Создание маски с помощью бинаризации и морфологических операций.
    """
    # 1. Переводим в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Размытие для удаления шумов
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. Бинаризация (используем метод Otsu для автоматического поиска порога)
    # Если фон светлый, а объект темный, используйте cv2.THRESH_BINARY_INV
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Морфологические преобразования
    # Создаем ядро (структурирующий элемент)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Закрытие (Closing) - удаляет черные дыры внутри белого объекта
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Открытие (Opening) - удаляет белый шум на фоне
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def create_grabcut_mask(img):
    """
    Сегментация объекта с помощью метода GrabCut.
    """
    mask = np.zeros(img.shape[:2], np.uint8)

    # Внутренние массивы, необходимые для работы GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Задаем прямоугольник, внутри которого находится объект (x, y, w, h)
    # По умолчанию берем центр изображения с отступом 10%
    h, w = img.shape[:2]
    margin_x, margin_y = int(w * 0.1), int(h * 0.1)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    # Запускаем алгоритм (5 итераций)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # GrabCut помечает пиксели 4 флагами.
    # Пиксели фона (0) и возможного фона (2) делаем черными (0).
    # Пиксели объекта (1) и возможного объекта (3) делаем белыми (25).
    mask_final = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")

    return mask_final


# --- Запуск Части 1 ---
if __name__ == "__main__":
    # Загружаем изображение
    image = cv2.imread("LAB_6/input.jpg")

    if image is None:
        print("Ошибка: Изображение 'input.jpg' не найдено!")
        exit()

    # Изменяем размер для удобства отображения, если оно слишком большое
    h, w = image.shape[:2]
    if w > 1000:
        image = cv2.resize(image, (800, int(800 * h / w)))

    # Получаем маски
    morph_mask = create_morphology_mask(image)
    grabcut_mask = create_grabcut_mask(image)

    # Сохраняем лучшую маску как базовую для следующей части (аугментации)
    # Обычно GrabCut дает более точный результат на сложных фото
    cv2.imwrite("LAB_6/base_mask.png", grabcut_mask)

    # Отображаем результаты для сравнения
    cv2.imshow("Original Image", image)
    cv2.imshow("Morphology Mask", morph_mask)
    cv2.imshow("GrabCut Mask", grabcut_mask)

    print("Нажмите любую клавишу в окне изображения для продолжения...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
