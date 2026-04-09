import sys
import random
import time
import re
import ast
from itertools import product
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QGroupBox,
                             QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QLabel, QComboBox,
                             QLineEdit, QPushButton, QTextEdit, QRadioButton, QButtonGroup,
                             QMessageBox, QDialog, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor, QLinearGradient, QBrush

# ─────────────────────────────────────────────────────────────
# 🎨 СТИЛИ (QSS) - Белый, Мятный, Бирюзовый с градиентами
# ─────────────────────────────────────────────────────────────
STYLESHEET = """
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                stop:0 #e0f7fa, stop:0.5 #e8f5e9, stop:1 #e0f2f1);
}
QTabWidget::pane {
    border: 1px solid #80cbc4;
    background: rgba(255, 255, 255, 0.75);
    border-radius: 12px;
    padding: 8px;
}
QTabBar::tab {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #b2dfdb, stop:1 #80cbc4);
    color: #004d40;
    padding: 12px 24px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    margin-right: 4px;
    font-weight: bold;
    border: 1px solid #4db6ac;
    border-bottom: none;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #00796b;
    border-bottom: 2px solid #009688;
}
QTabBar::tab:!selected {
    margin-top: 6px;
}
QGroupBox {
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid #80cbc4;
    border-radius: 10px;
    margin-top: 10px;
    padding: 15px;
    font-weight: bold;
    color: #004d40;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    padding: 0 8px;
    color: #00796b;
}
QLabel { color: #004d40; font-size: 14px; }
QComboBox, QLineEdit {
    background: #ffffff;
    border: 1px solid #80cbc4;
    border-radius: 6px;
    padding: 8px 12px;
    color: #006064;
    font-size: 13px;
}
QComboBox::drop-down, QLineEdit:focus {
    border: 1px solid #4db6ac;
    background: #f1f8e9;
}
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #26a69a, stop:1 #00897b);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 18px;
    font-weight: bold;
    font-size: 13px;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #009688, stop:1 #00796b);
}
QPushButton:pressed {
    background: #004d40;
}
QPushButton:disabled {
    background: #b0bec5;
    color: #ffffff;
}
QRadioButton {
    color: #004d40;
    spacing: 8px;
    padding: 4px;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #4db6ac;
    border-radius: 8px;
    background: #ffffff;
}
QRadioButton::indicator:checked {
    background: #009688;
    border: 2px solid #00796b;
}
QTextEdit, QPlainTextEdit {
    background: #ffffff;
    border: 1px solid #80cbc4;
    border-radius: 8px;
    color: #006064;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    padding: 8px;
}
"""


# ─────────────────────────────────────────────────────────────
# 🧮 ЯДРО: Тензоры и методы умножения
# ─────────────────────────────────────────────────────────────
class Tensor:
    def __init__(self, dimension, data=None):
        self.dimension = dimension
        self.data = data if data is not None else {}

    def add_value(self, indices, value):
        self.data[tuple(indices)] = value

    def get_value(self, indices):
        return self.data.get(tuple(indices), 0)

    def get_all_indices(self):
        return list(self.data.keys())

    def to_nested_list(self):
        if not self.data:
            return []
        shape = []
        for indices in self.data.keys():
            for i, idx in enumerate(indices):
                if i >= len(shape):
                    shape.append(idx + 1)
                else:
                    shape[i] = max(shape[i], idx + 1)
        result = self._create_nested_list(shape)
        for indices, value in self.data.items():
            self._set_value_in_list(result, indices, value)
        return result

    def _create_nested_list(self, shape):
        if len(shape) == 1:
            return [0.0] * shape[0]
        return [self._create_nested_list(shape[1:]) for _ in range(shape[0])]

    def _set_value_in_list(self, lst, indices, value):
        if len(indices) == 1:
            lst[indices[0]] = value
        else:
            self._set_value_in_list(lst[indices[0]], indices[1:], value)

    def get_shape(self):
        if not self.data:
            return ()
        shape = []
        for indices in self.data.keys():
            for i, idx in enumerate(indices):
                if i >= len(shape):
                    shape.append(idx + 1)
                else:
                    shape[i] = max(shape[i], idx + 1)
        return tuple(shape)

    @classmethod
    def from_nested_list(cls, nested_list):
        data = {}
        dimension = cls._get_dimension(nested_list)
        cls._fill_data_from_list(nested_list, (), data)
        return cls(dimension, data)

    @staticmethod
    def _get_dimension(lst):
        if not isinstance(lst, list) or len(lst) == 0:
            return 0
        return 1 + Tensor._get_dimension(lst[0])

    @staticmethod
    def _fill_data_from_list(lst, current_indices, data):
        if isinstance(lst, list):
            for i, item in enumerate(lst):
                Tensor._fill_data_from_list(item, current_indices + (i,), data)
        else:
            data[current_indices] = lst


class MunermanTensorMultiplier:
    @staticmethod
    def method1_cayley_square(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3 = key_b
                if i2 == j1 and i3 == j2:
                    new_key = (i1, j3)
                    result_data[new_key] = result_data.get(new_key, 0) + value_a * value_b
        return Tensor(2, result_data)

    @staticmethod
    def method1_cayley_4d(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3, i4 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3, j4 = key_b
                if i3 == j1 and i4 == j2:
                    new_key = (i1, i2, j3, j4)
                    result_data[new_key] = result_data.get(new_key, 0) + value_a * value_b
        return Tensor(4, result_data)

    @staticmethod
    def method1_cayley_3d_4d(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3, j4 = key_b
                if i2 == j1 and i3 == j2:
                    new_key = (i1, j3, j4)
                    result_data[new_key] = result_data.get(new_key, 0) + value_a * value_b
        return Tensor(3, result_data)

    @staticmethod
    def method1_cayley_4d_3d(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3, i4 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3 = key_b
                if i3 == j1 and i4 == j2:
                    new_key = (i1, i2, j3)
                    result_data[new_key] = result_data.get(new_key, 0) + value_a * value_b
        return Tensor(3, result_data)

    @staticmethod
    def method2_cayley_square(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3 = key_b
                if i3 == j1:
                    new_key = (i1, i2, j2, j3)
                    result_data[new_key] = result_data.get(new_key, 0) + value_a * value_b
        return Tensor(4, result_data)

    @staticmethod
    def method2_cayley_4d(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3, i4 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3, j4 = key_b
                if i4 == j1:
                    new_key = (i1, i2, i3, j2, j3, j4)
                    result_data[new_key] = result_data.get(new_key, 0) + value_a * value_b
        return Tensor(6, result_data)

    @staticmethod
    def method2_cayley_3d_4d(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3, j4 = key_b
                if i3 == j1:
                    new_key = (i1, i2, j2, j3, j4)
                    result_data[new_key] = result_data.get(new_key, 0) + value_a * value_b
        return Tensor(5, result_data)

    @staticmethod
    def method2_cayley_4d_3d(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3, i4 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3 = key_b
                if i4 == j1:
                    new_key = (i1, i2, i3, j2, j3)
                    result_data[new_key] = result_data.get(new_key, 0) + value_a * value_b
        return Tensor(5, result_data)

    @staticmethod
    def method3_scott_square(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3 = key_b
                if i2 == j1 and i3 == j2:
                    new_key = (i1, i2, i3, j3)
                    result_data[new_key] = value_a * value_b
        return Tensor(4, result_data)

    @staticmethod
    def method3_scott_4d(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3, i4 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3, j4 = key_b
                if i3 == j1 and i4 == j2:
                    new_key = (i1, i2, i3, i4, j3, j4)
                    result_data[new_key] = value_a * value_b
        return Tensor(6, result_data)

    @staticmethod
    def method3_scott_3d_4d(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3, j4 = key_b
                if i2 == j1 and i3 == j2:
                    new_key = (i1, i2, i3, j3, j4)
                    result_data[new_key] = value_a * value_b
        return Tensor(5, result_data)

    @staticmethod
    def method3_scott_4d_3d(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3, i4 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3 = key_b
                if i3 == j1 and i4 == j2:
                    new_key = (i1, i2, i3, i4, j3)
                    result_data[new_key] = value_a * value_b
        return Tensor(5, result_data)

    @staticmethod
    def method4_scott_square(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3 = key_b
                if i3 == j1:
                    new_key = (i1, i2, i3, j2, j3)
                    result_data[new_key] = value_a * value_b
        return Tensor(5, result_data)

    @staticmethod
    def method4_scott_4d(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3, i4 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3, j4 = key_b
                if i4 == j1:
                    new_key = (i1, i2, i3, i4, j2, j3, j4)
                    result_data[new_key] = value_a * value_b
        return Tensor(7, result_data)

    @staticmethod
    def method4_scott_3d_4d(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3, j4 = key_b
                if i3 == j1:
                    new_key = (i1, i2, i3, j2, j3, j4)
                    result_data[new_key] = value_a * value_b
        return Tensor(6, result_data)

    @staticmethod
    def method4_scott_4d_3d(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3, i4 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3 = key_b
                if i4 == j1:
                    new_key = (i1, i2, i3, i4, j2, j3)
                    result_data[new_key] = value_a * value_b
        return Tensor(6, result_data)

    @staticmethod
    def method5_combined_square(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3 = key_b
                if i3 == j1 and i2 == j2:
                    new_key = (i1, i2, j3)
                    result_data[new_key] = result_data.get(new_key, 0) + value_a * value_b
        return Tensor(3, result_data)

    @staticmethod
    def method5_combined_4d(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3, i4 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3, j4 = key_b
                if i4 == j1 and i3 == j2:
                    new_key = (i1, i2, i3, j3, j4)
                    result_data[new_key] = result_data.get(new_key, 0) + value_a * value_b
        return Tensor(5, result_data)

    @staticmethod
    def method5_combined_3d_4d(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3, j4 = key_b
                if i3 == j1 and i2 == j2:
                    new_key = (i1, i2, j3, j4)
                    result_data[new_key] = result_data.get(new_key, 0) + value_a * value_b
        return Tensor(4, result_data)

    @staticmethod
    def method5_combined_4d_3d(tensor_a, tensor_b):
        result_data = {}
        for key_a, value_a in tensor_a.data.items():
            i1, i2, i3, i4 = key_a
            for key_b, value_b in tensor_b.data.items():
                j1, j2, j3 = key_b
                if i4 == j1 and i3 == j2:
                    new_key = (i1, i2, i3, j3)
                    result_data[new_key] = result_data.get(new_key, 0) + value_a * value_b
        return Tensor(4, result_data)


class TensorOperations:
    @staticmethod
    def validate_shapes(tensor_a, tensor_b, method, dimension_type):
        shape_a = tensor_a.get_shape()
        shape_b = tensor_b.get_shape()

        if not shape_a or not shape_b:
            raise ValueError("Одна или обе матрицы пусты. Заполните данные перед умножением.")

        def check(condition, error_msg):
            if not condition:
                raise ValueError(error_msg)

        # Методы 1 и 3: свёртка по 2 индексам (предпоследний/последний у А -> первые два у Б)
        if method in (1, 3):
            if dimension_type in ('square', '3d_4d'):
                check(shape_a[1] == shape_b[0] and shape_a[2] == shape_b[1],
                      f"Метод {method}: A.shape[1]({shape_a[1]}) ≠ B.shape[0]({shape_b[0]}) или "
                      f"A.shape[2]({shape_a[2]}) ≠ B.shape[1]({shape_b[1]})")
            elif dimension_type in ('4d', '4d_3d'):
                check(shape_a[2] == shape_b[0] and shape_a[3] == shape_b[1],
                      f"Метод {method}: A.shape[2]({shape_a[2]}) ≠ B.shape[0]({shape_b[0]}) или "
                      f"A.shape[3]({shape_a[3]}) ≠ B.shape[1]({shape_b[1]})")

        # Методы 2 и 4: свёртка по 1 индексу (последний у А -> первый у Б)
        elif method in (2, 4):
            if dimension_type in ('square', '3d_4d'):
                check(shape_a[2] == shape_b[0],
                      f"Метод {method}: A.shape[2]({shape_a[2]}) ≠ B.shape[0]({shape_b[0]})")
            elif dimension_type in ('4d', '4d_3d'):
                check(shape_a[3] == shape_b[0],
                      f"Метод {method}: A.shape[3]({shape_a[3]}) ≠ B.shape[0]({shape_b[0]})")

        # Метод 5: смешанные индексы (последний у А -> первый у Б, предпоследний у А -> второй у Б)
        elif method == 5:
            if dimension_type in ('square', '3d_4d'):
                check(shape_a[2] == shape_b[0] and shape_a[1] == shape_b[1],
                      f"Метод {method}: A.shape[2]({shape_a[2]}) ≠ B.shape[0]({shape_b[0]}) или "
                      f"A.shape[1]({shape_a[1]}) ≠ B.shape[1]({shape_b[1]})")
            elif dimension_type in ('4d', '4d_3d'):
                check(shape_a[3] == shape_b[0] and shape_a[2] == shape_b[1],
                      f"Метод {method}: A.shape[3]({shape_a[3]}) ≠ B.shape[0]({shape_b[0]}) или "
                      f"A.shape[2]({shape_a[2]}) ≠ B.shape[1]({shape_b[1]})")

    @staticmethod
    def multiply_tensors(tensor_a, tensor_b, method, dimension_type):
        # 1. Жёсткая проверка совместимости перед вычислениями
        TensorOperations.validate_shapes(tensor_a, tensor_b, method, dimension_type)

        # 2. Выполнение умножения
        multiplier = MunermanTensorMultiplier()
        if method == 1:
            return getattr(multiplier, f'method1_cayley_{dimension_type}')(tensor_a, tensor_b)
        elif method == 2:
            return getattr(multiplier, f'method2_cayley_{dimension_type}')(tensor_a, tensor_b)
        elif method == 3:
            return getattr(multiplier, f'method3_scott_{dimension_type}')(tensor_a, tensor_b)
        elif method == 4:
            return getattr(multiplier, f'method4_scott_{dimension_type}')(tensor_a, tensor_b)
        elif method == 5:
            return getattr(multiplier, f'method5_combined_{dimension_type}')(tensor_a, tensor_b)
        else:
            raise ValueError(f"Неизвестный метод: {method}")


# ─────────────────────────────────────────────────────────────
# 🖥️ GUI: Редактор матриц
# ─────────────────────────────────────────────────────────────
class MatrixEditorDialog(QDialog):
    tensor_ready = pyqtSignal(object)

    def __init__(self, parent, matrix_type):
        super().__init__(parent)
        self.matrix_type = matrix_type
        self.setWindowTitle(f"Редактор матрицы {matrix_type}")
        self.resize(650, 450)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        info_lbl = QLabel(f"Введите матрицу {self.matrix_type} в формате Соколова:")
        layout.addWidget(info_lbl)

        self.text_area = QTextEdit()
        self.text_area.setPlaceholderText(f"[[[1, 2]; [3, 4]], [[5, 6]; [7, 8]]]")
        layout.addWidget(self.text_area, 1)

        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("💾 Сохранить")
        self.btn_cancel = QPushButton("❌ Отмена")
        self.btn_clear = QPushButton("🗑️ Очистить")
        self.btn_paste = QPushButton("📋 Вставить")

        for btn in [self.btn_paste, self.btn_clear, self.btn_cancel, self.btn_save]:
            btn_layout.addWidget(btn)
            if btn == self.btn_save:
                btn.clicked.connect(self.save_matrix)
            elif btn == self.btn_cancel:
                btn.clicked.connect(self.reject)
            elif btn == self.btn_clear:
                btn.clicked.connect(self._clear_text)
            elif btn == self.btn_paste:
                btn.clicked.connect(self._paste_from_clipboard)

        layout.addLayout(btn_layout)
        self._apply_styles()

    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog { background: #f0fdf4; }
            QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #26a69a, stop:1 #00897b);
                          color: white; border-radius: 6px; padding: 8px; font-weight: bold; }
            QPushButton:hover { background: #00796b; }
            QPushButton:pressed { background: #004d40; }
        """)

    def _clear_text(self):
        self.text_area.clear()

    def _paste_from_clipboard(self):
        self.text_area.paste()

    def save_matrix(self):
        try:
            text = self.text_area.toPlainText().strip()
            if not text:
                QMessageBox.warning(self, "Ошибка", "Поле ввода пустое!")
                return

            lines = [re.sub(r'#.*$', '', line).strip() for line in text.split('\n')]
            text = ' '.join(lines)
            text = re.sub(r'(\d),(\d)', r'\1.\2', text)
            text = self.convert_sokolov_to_python(text)

            matrix = ast.literal_eval(text)
            if not isinstance(matrix, list):
                raise ValueError("Введенные данные не являются матрицей")

            tensor = Tensor.from_nested_list(matrix)
            self.tensor_ready.emit(tensor)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка формата", f"Неверный синтаксис:\n{str(e)}")

    def convert_sokolov_to_python(self, text):
        if text.startswith('[') and text.endswith(']'):
            depth, result, i = 0, [], 0
            while i < len(text):
                char = text[i]
                if char == '[':
                    depth += 1
                    result.append(char)
                elif char == ']':
                    depth -= 1
                    result.append(char)
                elif char == ';' and depth == 1:
                    result.append('], [')
                elif char == ';':
                    result.append(',')
                else:
                    result.append(char)
                i += 1
            return ''.join(result)
        return text


# ─────────────────────────────────────────────────────────────
# 🖥️ GUI: Главное окно
# ─────────────────────────────────────────────────────────────
class MatrixAppPyQt6(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tensor_a = None
        self.tensor_b = None
        self.result_tensor = None
        self.setWindowTitle("Многомерные матрицы Спиридонова")
        self.resize(1000, 700)
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_creation_tab(), "🛠️ Создание матриц")
        self.tabs.addTab(self.build_multiplication_tab(), "✖️ Умножение матриц")
        self.tabs.addTab(self.build_results_tab(), "📊 Результаты")

        main_layout.addWidget(self.tabs)
        self.setStyleSheet(STYLESHEET)

    # ─── Вкладка 1: Создание ───
    def build_creation_tab(self):
        tab = QWidget()
        grid = QGridLayout(tab)
        grid.setSpacing(15)

        # Матрица A
        grp_a = QGroupBox("Матрица A")
        layout_a = QFormLayout()
        self.dim_a = QComboBox()
        self.dim_a.addItems(["3D", "4D"])
        self.shape_a = QLineEdit("2, 2, 2")

        layout_a.addRow("Размерность:", self.dim_a)
        layout_a.addRow("Форма (через запятую):", self.shape_a)

        btn_a_rand = QPushButton("🎲 Создать случайную")
        btn_a_manual = QPushButton("✏️ Ввести вручную")
        btn_a_show = QPushButton("👁️ Показать полную")
        btn_a_rand.clicked.connect(lambda: self.create_random_tensor('A'))
        btn_a_manual.clicked.connect(lambda: self.open_matrix_editor('A'))
        btn_a_show.clicked.connect(lambda: self.show_full_tensor('A'))

        layout_a.addRow(btn_a_rand)
        layout_a.addRow(btn_a_manual)
        layout_a.addRow(btn_a_show)
        self.lbl_info_a = QLabel("Матрица A не создана")
        layout_a.addRow(self.lbl_info_a)
        grp_a.setLayout(layout_a)

        # Матрица B
        grp_b = QGroupBox("Матрица B")
        layout_b = QFormLayout()
        self.dim_b = QComboBox()
        self.dim_b.addItems(["3D", "4D"])
        self.shape_b = QLineEdit("2, 2, 2")

        layout_b.addRow("Размерность:", self.dim_b)
        layout_b.addRow("Форма (через запятую):", self.shape_b)

        btn_b_rand = QPushButton("🎲 Создать случайную")
        btn_b_manual = QPushButton("✏️ Ввести вручную")
        btn_b_show = QPushButton("👁️ Показать полную")
        btn_b_rand.clicked.connect(lambda: self.create_random_tensor('B'))
        btn_b_manual.clicked.connect(lambda: self.open_matrix_editor('B'))
        btn_b_show.clicked.connect(lambda: self.show_full_tensor('B'))

        layout_b.addRow(btn_b_rand)
        layout_b.addRow(btn_b_manual)
        layout_b.addRow(btn_b_show)
        self.lbl_info_b = QLabel("Матрица B не создана")
        layout_b.addRow(self.lbl_info_b)
        grp_b.setLayout(layout_b)

        grid.addWidget(grp_a, 0, 0)
        grid.addWidget(grp_b, 0, 1)

        # Лог
        grp_log = QGroupBox("📜 Журнал операций")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.append("Добро пожаловать в приложение!\nИспользуйте вкладки для работы с матрицами.")
        grp_log.setLayout(QVBoxLayout())
        grp_log.layout().addWidget(self.log_text)
        grid.addWidget(grp_log, 1, 0, 1, 2)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        return tab

    # ─── Вкладка 2: Умножение ───
    def build_multiplication_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        lbl_method = QLabel("Выберите метод умножения:", font=QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(lbl_method)

        method_group = QGroupBox("Алгоритм свертки")
        grid = QGridLayout(method_group)
        self.method_var = QButtonGroup(self)

        methods = [
            ("1. (0,2)-свернутое произведение - Кэлиевы индексы", 1),
            ("2. (0,1)-свернутое произведение - Кэлиев индекс", 2),
            ("3. (2,0)-свернутое произведение - Скоттовы индексы", 3),
            ("4. (1,0)-свернутое произведение - Скоттов индекс", 4),
            ("5. (1,1)-свернутое произведение - Смешанные индексы", 5)
        ]
        for i, (txt, val) in enumerate(methods):
            rb = QRadioButton(txt)
            rb.setChecked(val == 1)
            self.method_var.addButton(rb, val)
            grid.addWidget(rb, i, 0)
        layout.addWidget(method_group)

        self.btn_multiply = QPushButton("🚀 Выполнить умножение")
        self.btn_multiply.clicked.connect(self.perform_multiplication)
        self.btn_multiply.setMinimumHeight(40)
        layout.addWidget(self.btn_multiply, alignment=Qt.AlignmentFlag.AlignCenter)

        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_status)

        return tab

    # ─── Вкладка 3: Результаты ───
    def build_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.res_text = QTextEdit()
        self.res_text.setReadOnly(True)
        layout.addWidget(self.res_text)
        btn_clear = QPushButton("🗑️ Очистить результаты")
        btn_clear.clicked.connect(self.clear_results)
        layout.addWidget(btn_clear, alignment=Qt.AlignmentFlag.AlignRight)
        return tab

    # ─── Логика ───
    def create_random_tensor(self, tensor_type):
        try:
            dim = self.dim_a.currentText() if tensor_type == 'A' else self.dim_b.currentText()
            shape_str = self.shape_a.text() if tensor_type == 'A' else self.shape_b.text()
            shape = tuple(map(int, shape_str.replace(' ', '').split(',')))

            if dim == "3D" and len(shape) != 3:
                raise ValueError("Для 3D матрицы нужно 3 размера")
            if dim == "4D" and len(shape) != 4:
                raise ValueError("Для 4D матрицы нужно 4 размера")

            tensor = self._generate_random_tensor(shape)
            if tensor_type == 'A':
                self.tensor_a = tensor
                shape_disp = "×".join(map(str, tensor.get_shape()))
                self.lbl_info_a.setText(f"✅ Матрица A: {shape_disp}")
            else:
                self.tensor_b = tensor
                shape_disp = "×".join(map(str, tensor.get_shape()))
                self.lbl_info_b.setText(f"✅ Матрица B: {shape_disp}")

            self.log(f"Создана случайная матрица {tensor_type} формы {shape_disp}")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка параметров", str(e))

    def _generate_random_tensor(self, shape):
        tensor = Tensor(len(shape))
        for idx in product(*[range(d) for d in shape]):
            tensor.add_value(idx, round(random.uniform(0, 10), 2))
        return tensor

    def open_matrix_editor(self, tensor_type):
        editor = MatrixEditorDialog(self, tensor_type)
        editor.tensor_ready.connect(lambda t: self.set_tensor_from_editor(tensor_type, t))
        editor.exec()

    def set_tensor_from_editor(self, tensor_type, tensor):
        shape_disp = "×".join(map(str, tensor.get_shape()))
        if tensor_type == 'A':
            self.tensor_a = tensor
            self.lbl_info_a.setText(f"✅ Матрица A: {shape_disp}")
        else:
            self.tensor_b = tensor
            self.lbl_info_b.setText(f"✅ Матрица B: {shape_disp}")
        self.log(f"Матрица {tensor_type} загружена вручную ({shape_disp})")

    def show_full_tensor(self, tensor_type):
        tensor = self.tensor_a if tensor_type == 'A' else self.tensor_b
        if tensor is None:
            QMessageBox.warning(self, "Предупреждение", f"Матрица {tensor_type} не создана!")
            return

        txt = self.matrix_to_string_sokolov(tensor.to_nested_list())
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Полный просмотр матрицы {tensor_type}")
        dlg.resize(600, 500)
        layout = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setText(f"Матрица {tensor_type}:\n\n{txt}")
        layout.addWidget(te)
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)
        dlg.exec()

    def matrix_to_string_sokolov(self, matrix):
        if isinstance(matrix, (int, float)):
            return f"{matrix:.2f}"
        if isinstance(matrix, list):
            if not matrix:
                return "[]"
            dim = self._get_tensor_dimension(matrix)
            if dim == 1:
                return "[ " + ", ".join(f"{x:.2f}" for x in matrix) + " ]"
            if dim == 2:
                rows = []
                for row in matrix:
                    if isinstance(row, list):
                        rows.append("[ " + ", ".join(f"{x:.2f}" for x in row) + " ]")
                    else:
                        rows.append(f"{row:.2f}")
                return "[ " + "; ".join(rows) + " ]"
            return self._format_nd_matrix_sokolov(matrix, dim)
        return str(matrix)

    def _format_nd_matrix_sokolov(self, matrix, dim, level=0):
        if not isinstance(matrix, list) or not matrix:
            return "[]"
        if level == dim - 1:
            return "[ " + ", ".join(
                f"{x:.2f}" if isinstance(x, (int, float)) else self._format_nd_matrix_sokolov(x, dim, level + 1) for x
                in matrix) + " ]"
        if level == dim - 2:
            return "[ " + "; ".join(self._format_nd_matrix_sokolov(x, dim, level + 1) for x in matrix) + " ]"
        indent = "   " * (level + 1)
        parts = [self._format_nd_matrix_sokolov(x, dim, level + 1) for x in matrix]
        return "[\n" + indent + (f"\n{indent}").join(parts) + "\n" + "   " * level + "]"

    def _get_tensor_dimension(self, tensor):
        if not isinstance(tensor, list) or not tensor:
            return 0
        return 1 + self._get_tensor_dimension(tensor[0])

    def perform_multiplication(self):
        if self.tensor_a is None or self.tensor_b is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала создайте обе матрицы!")
            return

        method = self.method_var.checkedId()
        dim_a, dim_b = self.tensor_a.dimension, self.tensor_b.dimension
        if dim_a == 3 and dim_b == 3:
            dim_type = 'square'
        elif dim_a == 4 and dim_b == 4:
            dim_type = '4d'
        elif dim_a == 3 and dim_b == 4:
            dim_type = '3d_4d'
        elif dim_a == 4 and dim_b == 3:
            dim_type = '4d_3d'
        else:
            QMessageBox.critical(self, "Ошибка", f"Неподдерживаемые размерности: A={dim_a}D, B={dim_b}D")
            return

        try:
            self.btn_multiply.setEnabled(False)
            self.lbl_status.setText("⏳ Вычисление...")
            QApplication.processEvents()

            start = time.time()
            self.result_tensor = TensorOperations.multiply_tensors(self.tensor_a, self.tensor_b, method, dim_type)
            elapsed = time.time() - start

            res_shape = self.result_tensor.get_shape()
            self.lbl_status.setText(f"✅ Готово за {elapsed:.6f} сек | Результат: {res_shape}")
            self.log(f"Метод {method} выполнен за {elapsed:.6f} сек. Результат: {res_shape}")

            out = f"=== РЕЗУЛЬТАТЫ УМНОЖЕНИЯ ===\n\nМетод: {method}\nВремя: {elapsed:.6f} сек\nФорма: {res_shape}\n\nРезультат:\n"
            out += self.matrix_to_string_sokolov(self.result_tensor.to_nested_list())
            self.res_text.setText(out)
            self.tabs.setCurrentIndex(2)

        except ValueError as ve:
            # Специальная обработка для несовпадения размерностей
            QMessageBox.warning(self, "❌ Ошибка размерностей", str(ve))
            self.log(f"⚠️ Ошибка совместимости индексов: {ve}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка вычислений", str(e))
            self.log(f"❌ Ошибка в методе {method}: {e}")
        finally:
            self.btn_multiply.setEnabled(True)

    def clear_results(self):
        self.res_text.clear()

    def log(self, msg):
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())


# ─────────────────────────────────────────────────────────────
# 🚀 Запуск
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MatrixAppPyQt6()
    window.show()
    sys.exit(app.exec())