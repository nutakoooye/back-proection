from numba import cuda
import numpy as np


@cuda.jit(device=True)
def dot_matrix_cuda(A, B, result):
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i, j] += A[i, k] * B[k, j]

@cuda.jit(device=True)
def inverse_matrix_cuda3(matrix, inverse):
    # Получение размера матрицы
    n = 3

    # Проверка, что матрица квадратная
    if n != matrix.shape[1]:
        raise ValueError("Матрица должна быть квадратной")

    # Создание расширенной матрицы, добавляя единичную матрицу справа
    augmented_matrix = cuda.local.array((3, 6), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            augmented_matrix[i, j] = matrix[i, j]
            augmented_matrix[i, j + n] = 1 if i == j else 0

    # Прямой ход метода Гаусса
    for i in range(n):
        # Находим ведущий элемент
        pivot = augmented_matrix[i, i]

        # Делим текущую строку на ведущий элемент
        for j in range(2 * n):
            augmented_matrix[i, j] /= pivot

        # Обнуляем остальные элементы в столбце
        for j in range(n):
            if j != i:
                factor = augmented_matrix[j, i]
                for k in range(2 * n):
                    augmented_matrix[j, k] -= factor * augmented_matrix[i, k]

    # Извлекаем обратную матрицу из расширенной матрицы
    for i in range(n):
        for j in range(n):
            inverse[i, j] = augmented_matrix[i, j + n]

    # Извлекаем обратную матрицу из расширенной матрицы
    for i in range(n):
        for j in range(n):
            inverse[i, j] = augmented_matrix[i, j + n]


@cuda.jit(device=True)
def inverse_matrix_cuda4(matrix, inverse):
    # Получение размера матрицы
    n = 4

    # Проверка, что матрица квадратная
    if n != matrix.shape[1]:
        raise ValueError("Матрица должна быть квадратной")

    # Создание расширенной матрицы, добавляя единичную матрицу справа
    augmented_matrix = cuda.local.array((4, 8), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            augmented_matrix[i, j] = matrix[i, j]
            augmented_matrix[i, j + n] = 1 if i == j else 0

    # Прямой ход метода Гаусса
    for i in range(n):
        # Находим ведущий элемент
        pivot = augmented_matrix[i, i]

        # Делим текущую строку на ведущий элемент
        for j in range(2 * n):
            augmented_matrix[i, j] /= pivot

        # Обнуляем остальные элементы в столбце
        for j in range(n):
            if j != i:
                factor = augmented_matrix[j, i]
                for k in range(2 * n):
                    augmented_matrix[j, k] -= factor * augmented_matrix[i, k]

    # Извлекаем обратную матрицу из расширенной матрицы
    for i in range(n):
        for j in range(n):
            inverse[i, j] = augmented_matrix[i, j + n]

    # Извлекаем обратную матрицу из расширенной матрицы
    for i in range(n):
        for j in range(n):
            inverse[i, j] = augmented_matrix[i, j + n]


@cuda.jit(device=True)
def flip_and_transpose_cuda(pr, pr1):
    rows, cols = pr.shape

    # Копирование элементов в обратном порядке
    for i in range(rows):
        for j in range(cols):
            pr1[j, i] = pr[rows - 1 - i, j]
