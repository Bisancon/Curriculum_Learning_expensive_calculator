
# Описание проекта

Этот проект направлен на исследование метода **Curriculum Learning** для улучшения обучения трансформеров на примере решения простых математических операций. Метод **Curriculum Learning** предполагает, что модель сначала обучается на простых примерах, а затем постепенно переходит к более сложным. Гипотеза заключается в том, что поэтапное обучение модели на примерах возрастающей сложности способствует более быстрой сходимости и улучшает итоговое качество, по сравнению с обучением, основанным на случайном порядке подачи данных. Это позволяет моделям быстрее освоить основные концепции, прежде чем столкнуться с более трудными задачами. В рамках эксперимента проводится два независимых этапа обучения. На первом этапе данные подаются в случайном порядке, что соответствует стандартному подходу. На втором этапе используется **Curriculum Learning**.

В рамках этого проекта используется **Transformer Decoder** для решения математических операций, таких как сложение, вычитание, умножение и деление. Процесс обучения прис использовании **Curriculum Learning** состоит из двух фаз:
1. **Обучение на простых данных**: Простейшие математические выражения с двумя числами.
2. **Обучение на сложных данных**: Простейшие математические выражения с количеством чисел от 3 до 4.

Мы проверяем гипотезу, что **Curriculum Learning** улучшит сходимость и качество модели, сравнивая результаты, полученные с различными стратегиями сэмплирования данных. Предполагается, что обучение с использованием Curriculum Learning приведёт к улучшению как качества, так и скорости обучения.

# Цели проекта

- Исследовать влияние подхода Curriculum Learning на задачу математических вычислений.
- Реализовать модель на основе трансформера для решения простых и сложных математических операций.
- Создать кастомный токенизатор для чисел и операторов в математических выражениях.
- Провести эксперимент с различными стратегиями сэмплирования данных (сложные -> простые и рандомизированные данные).
- Оценить результаты обучения и сходимость модели.
# Реализация
## Генерация данных

```bash
python data_generator.py
```

Для генерации данных используется скрипт data_generator.py, который генерирует два типа наборов данных:

Простые операции: генерируются арифметические выражения, содержащие два числа (например, "1 + 1" или "5 * 3"). Эти операции включают сложение, вычитание, умножение и деление.
Сложные операции: генерируются выражения с тремя и четырьмя числами (например, ({num1} + {num2}) * {num3} или ({num1} + {num2}) * ({num3} - {num4})). Эти операции позволяют создавать более сложные математические выражения, которые включают несколько чисел и различные арифметические действия(из того же набора что и в простых операциях).

Результаты генерации сохраняются в два файла:

curriculum_data.csv — содержит все сгенерированные данные с простыми и сложными операциями.
shuffled_data.csv — содержит те же данные, но в случайном порядке.
test_data.csv — тестовый набор данных с более сложными выражениями для проверки модели.


## Токенизатор
Токенизатор **CurriculumTokenizer** предназначен для преобразования текстовых выражений в последовательность токенов, которые могут быть использованы моделью для обучения. Он также предоставляет возможность декодировать последовательности токенов обратно в текстовые выражения. Функция **encode** токенизирует текстовое выражение, а **decode** преобразует токены в текст.

Основные токены:
<PAD>: токен для дополнения последовательностей до одинаковой длины.
<SOS>: токен начала последовательности.
<EOS>: токен конца последовательности.
<UNK>: токен для неизвестных символов.
Математические операторы: '+': 4, '-': 5, '*': 6, '^': 7, '=': 8.
Цифры: от 0 до 9, которым соответствуют токены от 10 до 19.

## Модель
Модель основана на Transformer Decoder. Входные данные представляют собой последовательности токенов, полученных из математических выражений, которые модель обрабатывает и предсказывает результаты операций.

Параметры модели:
num_tokens: количество уникальных токенов в словаре (включая операторы и цифры).
n_embd: размерность эмбеддингов для входных токенов.
num_layers: количество слоев декодера.
num_heads: количество голов в механизме внимания.
num_classes: количество классов, равное диапазону возможных результатов операций.
Обучение:
Обучение модели происходит в два этапа:

Обучение с случайным сэмплированием (Random Sampling): На этом этапе данные подаются в случайном порядке, что моделирует случайный порядок представления задач при обучении. Используется стандартный загрузчик данных и стандартный порядок обработки.
Обучение с Curriculum Learning (обучение с постепенным усложнением задач): Данные подаются в измененном порядке, начиная с простых примеров и постепенно переходя к более сложным. Такой подход помогает улучшить сходимость и качество обучения.


# Результаты эксперимента

![image](https://github.com/user-attachments/assets/b9f3e547-bd7d-4734-b8c8-1463092ca818)

На графиках представлены результаты эксперимента, которые показывают динамику **Loss** и **Accuracy** для обоих подходов к обучению

На первом графике видно, что потери на тренировочной выборке уменьшаются в обоих подходах по мере увеличения числа эпох. При этом обучение с использованием **Curriculum Learning** демонстрирует более быстрое снижение потерь на тренировочной выборке, начиная с низких значений уже на первых эпохах. **Val Loss** для метода **Curriculum Learning**, напротив, остаются значительно выше на протяжении всего эксперимента, что может указывать на недостаточную генерализацию модели. Для случайного сэмплирования валидационные потери ниже, но всё же демонстрируют тренд на их увеличение с течением времени.

Второй график показывает изменение точности на тренировочной и валидационной выборках. Точность на тренировочной выборке для метода **Curriculum Learning** быстро растёт и достигает порядка 80% к десятой эпохе, превосходя случайное сэмплирование, которое достигает лишь порядка 60%. Однако точность на валидационной выборке для **Curriculum Learning** остаётся низкой и колеблется около 30–35%, практически не изменяясь в течение эксперимента. Для случайного сэмплирования валидационная точность демонстрирует схожий уровень с **Curriculum Learning**, но с меньшими колебаниями.

# Выводы

Результаты эксперимента демонстрируют, что использование **Curriculum Learning** способствует более быстрому снижению функции потерь и повышению точности на тренировочной выборке, что подтверждает гипотезу о его эффективности в улучшении сходимости модели. Однако на валидационной выборке данный подход не приводит к улучшению качества обобщения, о чём свидетельствует низкая и практически неизменная точность, а также высокие потери. Это может происходить по ряду причин, таких как переобучение и недостаточная генерализация. Модель могла обучиться специфическим паттернам в данных, предоставленных в определённом порядке, что ухудшает её способность работать с примерами, которые не следуют этому порядку. 
Таким образом, **Curriculum Learning** подтвердил свою эффективность в ускорении обучения модели, однако с точки зрения качества предсказаний на валидационных данных его результаты пока недостаточны для того что бы говорит о пользе данного подхода.

# Запуск эксперимента

Для запуска проекта, сначала клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/curriculum-learning-math-transformer.git
cd Curriculum_Learning_expensive_calculator
python3 -m venv venv
.\venv\Scripts\activate
pip freeze > requirements.txt
```


# Распределение

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

