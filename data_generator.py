import numpy as np
import pandas as pd
from random import shuffle

# Функция для генерации простых и сложных выражений с скобками и несколькими числами
def generate_complex_data():
    data = []

    # Примеры с однозначными и двузначными числами
    for i in range(0, 10, 5):  # i = 0 to 9
        for j in range(1, 10):  # j = 1 to 9
            i_random = i + np.random.randint(0, 5)  # Случайное прибавление к i
            j_random = j + np.random.randint(0, 5)  # Случайное прибавление к j

            # Простые операции
            expr = f"{i_random} + {j_random}"
            result = i_random + j_random
            if result >= 0:
                data.append({'expression': expr, 'result': result})

            expr = f"{i_random} - {j_random}"
            result = i_random - j_random
            if result >= 0:
                data.append({'expression': expr, 'result': result})

            expr = f"{i_random} * {j_random}"
            result = i_random * j_random
            data.append({'expression': expr, 'result': result})

            if j_random != 0 and i_random % j_random == 0:
                expr = f"{i_random} / {j_random}"
                result = i_random // j_random
                data.append({'expression': expr, 'result': result})

            expr = f"{i_random}^2"
            result = i_random ** 2
            data.append({'expression': expr, 'result': result})

    for i in range(10, 100, 5):  # i = 10 to 99
        for j in range(10, 100):  # j = 10 to 99
            i_random = i + np.random.randint(0, 5)  # Случайное прибавление к i
            j_random = j + np.random.randint(0, 5)  # Случайное прибавление к j

            expr = f"{i_random} + {j_random}"
            result = i_random + j_random
            if result >= 0:
                data.append({'expression': expr, 'result': result})

            expr = f"{i_random} - {j_random}"
            result = i_random - j_random
            if result >= 0:
                data.append({'expression': expr, 'result': result})

            expr = f"{i_random} * {j_random}"
            result = i_random * j_random
            data.append({'expression': expr, 'result': result})

            if j_random != 0 and i_random % j_random == 0:
                expr = f"{i_random} / {j_random}"
                result = i_random // j_random
                data.append({'expression': expr, 'result': result})

            expr = f"{i_random}^2"
            result = i_random ** 2
            data.append({'expression': expr, 'result': result})

    # Примеры с 3 и 4 числами (2000 примеров)
    for _ in range(1000):  # 1000 примеров с 3 числами
        numbers = np.random.randint(1, 100, 3)  # 3 случайных числа
        expr = f"({numbers[0]} + {numbers[1]}) * {numbers[2]}"
        result = (numbers[0] + numbers[1]) * numbers[2]
        if result >= 0:
            data.append({'expression': expr, 'result': result})

    for _ in range(1000):  # 1000 примеров с 4 числами
        numbers = np.random.randint(1, 100, 4)  # 4 случайных числа
        expr = f"({numbers[0]} + {numbers[1]}) * ({numbers[2]} - {numbers[3]})"
        result = (numbers[0] + numbers[1]) * (numbers[2] - numbers[3])
        if result >= 0:
            data.append({'expression': expr, 'result': result})

    return pd.DataFrame(data)

# Генерация данных
complex_data = generate_complex_data()

# Перемешивание для случайного порядка
shuffled_data = complex_data.sample(frac=1).reset_index(drop=True)

# Генерация тестового набора (сложные примеры)
def generate_test_data():
    test_data = []
    for _ in range(100):
        a = np.random.randint(1, 1000)
        b = np.random.randint(1, 1000)
        c = np.random.randint(1, 1000)
        d = np.random.randint(1, 1000)
        expr = f"({a} * {b}) / ({c} + {d})"
        result = (a * b) / (c + d)
        test_data.append({'expression': expr, 'result': result})
    return pd.DataFrame(test_data)

# Тестовый набор данных
test_data = generate_test_data()

# Сохранение данных
complex_data.to_csv('сurriculum_data.csv', index=False)
shuffled_data.to_csv('shuffled_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
