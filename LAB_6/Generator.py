import os
import random
import cv2
import numpy as np


class AugmentationGenerator:
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask

    def _rotate_bound(self, img, angle, is_mask=False):
        """
        Поворот с сохранением всего изображения (без обрезки углов).
        """
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # Новые размеры bounding box
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Корректируем матрицу сдвига
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        flags = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.warpAffine(
            img, M, (new_w, new_h), flags=flags, borderValue=(0, 0, 0)
        )

    def augment(self):
        """
        Применяет случайный набор трансформаций к паре (изображение, маска).
        """
        img = self.image.copy()
        mask = self.mask.copy()

        # 1. Отражение по горизонтали (вероятность 50%)
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # 2. Отражение по вертикали (вероятность 20%)
        if random.random() > 0.8:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)

        # 3. Поворот на случайный угол с сохранением границ
        angle = random.uniform(-45, 45)
        img = self._rotate_bound(img, angle, is_mask=False)
        mask = self._rotate_bound(mask, angle, is_mask=True)

        # 4. Смещение (Translation)
        h, w = img.shape[:2]
        tx = random.randint(-int(w * 0.1), int(w * 0.1))
        ty = random.randint(-int(h * 0.1), int(h * 0.1))
        M_shift = np.float32([[1, 0, tx], [0, 1, ty]])

        img = cv2.warpAffine(
            img, M_shift, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
        )
        mask = cv2.warpAffine(
            mask, M_shift, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
        )

        # 5. Масштабирование (Zoom in / Zoom out)
        scale = random.uniform(0.8, 1.2)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Возвращаем к исходному размеру через обрезку или добавление полей
        if scale > 1.0:  # Обрезаем центр
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            img = img[start_y : start_y + h, start_x : start_x + w]
            mask = mask[start_y : start_y + h, start_x : start_x + w]
        else:  # Добавляем черные поля
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            img = cv2.copyMakeBorder(
                img,
                pad_y,
                h - new_h - pad_y,
                pad_x,
                w - new_w - pad_x,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            mask = cv2.copyMakeBorder(
                mask,
                pad_y,
                h - new_h - pad_y,
                pad_x,
                w - new_w - pad_x,
                cv2.BORDER_CONSTANT,
                value=0,
            )

        # --- ТОЛЬКО ДЛЯ ИЗОБРАЖЕНИЯ (Цветовые фильтры и шумы) ---

        # 6. Изменение яркости и контрастности
        alpha = random.uniform(0.7, 1.3)  # Контраст
        beta = random.randint(-30, 30)  # Яркость
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # 7. Применение фильтра (Размытие) с вероятностью 30%
        if random.random() > 0.7:
            k_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k_size, k_size), 0)

        # Убедимся, что маска строго бинарная после всех трансформаций
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return img, mask


# --- Запуск генерации датасета ---
if __name__ == "__main__":
    # Создаем директорию для датасета
    output_dir = "LAB_6/dataset"
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем оригинал и полученную ранее маску
    base_img = cv2.imread("LAB_6/input.jpg")
    base_mask = cv2.imread("LAB_6/base_mask.png", cv2.IMREAD_GRAYSCALE)

    if base_img is None or base_mask is None:
        print(
            "Ошибка: Необходимы файлы 'input.jpg' и 'base_mask.png'. Сначала запустите Часть 1."
        )
        exit()

    # Приводим к одному размеру (на случай расхождений)
    h, w = base_img.shape[:2]
    base_mask = cv2.resize(base_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    generator = AugmentationGenerator(base_img, base_mask)
    num_samples = 40  # Требуется 30-50 изображений

    print(f"Генерация {num_samples} пар изображений...")

    for i in range(num_samples):
        aug_img, aug_mask = generator.augment()

        # Имена файлов
        img_name = os.path.join(output_dir, f"sample_{i:03d}_img.jpg")
        mask_name = os.path.join(output_dir, f"sample_{i:03d}_mask.png")

        # Сохранение
        cv2.imwrite(img_name, aug_img)
        cv2.imwrite(mask_name, aug_mask)

    print(f"Успешно! Датасет сохранен в папку '{output_dir}'.")
