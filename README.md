# Curriculum Learning и Трансформеры для Математических Операций

## Описание проекта

Этот проект направлен на исследование метода **Curriculum Learning** для улучшения обучения трансформеров при решении математических задач. Метод **Curriculum Learning** предполагает, что модель сначала обучается на простых примерах, а затем постепенно переходит к более сложным. Это позволяет моделям быстрее освоить основные концепции, прежде чем столкнуться с более трудными задачами.

В рамках этого проекта используется **Transformer Decoder** для решения математических операций, таких как сложение, вычитание, умножение и деление. Процесс обучения состоит из двух фаз:
1. **Обучение на простых данных**: Простейшие математические выражения (например, от 1 до 10).
2. **Обучение на сложных данных**: Математические выражения с большими числами (например, от 10 до 100).

Мы проверяем гипотезу, что **Curriculum Learning** улучшит сходимость и качество модели, сравнивая результаты, полученные с различными стратегиями сэмплирования данных.

## Цели проекта

- Исследовать влияние подхода Curriculum Learning на задачу математических вычислений.
- Реализовать модель на основе трансформера для решения простых и сложных математических операций.
- Создать кастомный токенизатор для чисел и операторов в математических выражениях.
- Провести эксперимент с различными стратегиями сэмплирования данных (сложные -> простые и рандомизированные данные).
- Оценить результаты обучения и сходимость модели.

## Установка

Для запуска проекта, сначала клонируйте репозиторий:

## Распределение

Человек 1: Подготовка данных
Задача: Сгенерировать наборы данных для математических операций.
Простые операции (например, сложение, вычитание, умножение).
Сложные операции (например, большие числа, сложные выражения).
Подготовить две версии данных: одну для обучения с случайным сэмплированием, другую для Curriculum Learning (сложные задачи сначала).
Оформить данные в формат, подходящий для обучения модели (например, CSV или JSON).

Человек 2: Разработка и настройка модели (Transformer Decoder)
Задача: Разработать модель на основе Transformer Decoder для решения математических операций.
Определить архитектуру модели: количество слоев, размерность скрытых состояний, количество голов в мультиголовном внимании.
Настроить основные гиперпараметры: скорость обучения, размер батча, количество эпох.
Настроить обучающий процесс и создать функцию потерь для задач регрессии или классификации, в зависимости от формата задачи.

Человек 3: Реализация обучающих процессов
Задача: Реализовать процесс обучения для двух подходов.
Случайное сэмплирование: обучение модели на случайных примерах (случайный порядок примеров).
Curriculum Learning: обучение с поэтапным усложнением (сложные задачи сначала, затем простые).
Реализовать разделение на два сценария обучения и протестировать модель с каждым подходом.
Параллельно следить за метками качества модели (например, средняя ошибка на валидации, метрики точности).

Человек 4: Оценка и анализ результатов
Задача: Оценить результаты модели для обоих подходов.
Рассчитать метрики качества (например, точность, средняя ошибка) для каждой модели.
Сравнить скорость сходимости (например, время до достижения определенной точности).
Построить графики, иллюстрирующие процесс обучения (например, график ошибок на тренировочных и валидационных данных).
Оценить общее поведение модели на тестовых данных.

Человек 5: Документация и подготовка отчета
Задача: Подготовить отчет по результатам эксперимента.
Описать методы и подходы (Curriculum Learning и случайное сэмплирование).
Описать архитектуру модели и этапы обучения.
Документировать полученные результаты, выводы и возможные объяснения (например, почему один метод оказался эффективнее другого).
Подготовить слайды для представления работы команде или заинтересованным сторонам.

```bash
git clone https://github.com/yourusername/curriculum-learning-math-transformer.git
cd Curriculum_Learning_expensive_calculator
python3 -m venv venv
.\venv\Scripts\activate
pip freeze > requirements.txt
```

## Реализация


## Генерация данных
Для генерации данных используется файл data_generator.py. Он генерирует два типа наборов данных:

```bash
python data_generator.py
```

Простые операции: Генерируются выражения с двумя числами (например, 1 + 1, 5 * 3).
Сложные операции: Генерируются выражения с тремя и четырьмя числами (например, ({num1} + {num2}) * {num3}, ({num1} + {num2}) * ({num3} - {num4})).
Результаты сохраняются в файлы сurriculum_data.csv, shuffled_data.csv (разбросанные случайным образом примеры из сurriculum_data). Тестовые данные сохраняются в test_data.csv.


## Токенизатор
encode токенизирует текстовое выражение:
<PAD>: 0, для дополнения последовательностей до одинаковой длины
<SOS>: 1, для указания начала последовательности.
<EOS>: 2, для обозначения конца последовательности
<UNK>: 3, для обработки неизвестных токенов
'+': 4, '-': 5, '*': 6, '^': 7, '=': 8,
Далее цифры от 0 до 9: 10-19

decode преобразует токены в текст