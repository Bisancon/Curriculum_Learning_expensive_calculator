# Класс токенизатора
class CurriculumTokenizer:
    def __init__(self):
        # Базовые токены
        self.vocab = {
            '<PAD>': 0, # Для дополнения последовательностей до одинаковой длины
            '<SOS>': 1, # Для указания начала последовательности.
            '<EOS>': 2, # Для обозначения конца последовательности
            '<UNK>': 3, # Для обработки неизвестных токенов (out-of-vocabulary
            '+': 4,
            '-': 5,
            '*': 6,
            '^': 7,
            '=': 8,
            # '%': 9, # Его не используем
        }
        # Добавляем цифры с токенами от 10 до 19
        for i in range(10):
            self.vocab[str(i)] = 10 + i

        # Обратный словарь для преобразования токенов в символы
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    # Преобразование теста выражения в токены
    # Числа токенизируются как цифры
    def encode(self, expression):
        tokens = []
        tokens.append(self.vocab['<SOS>'])  # Начало последовательности
        for char in expression:
            if char in self.vocab:
                tokens.append(self.vocab[char])  # Преобразование символа в токен
            else:
                tokens.append(self.vocab['<UNK>'])  # Токен для неизвестного символа
        tokens.append(self.vocab['<EOS>'])  # Конец последовательности
        return tokens

    # Преобразование токенов обратно в текст выражения
    def decode(self, tokens):
        return ''.join(self.reverse_vocab[token] for token in tokens)

# Пример использования
tokenizer = CurriculumTokenizer()

example = "35-14=21"
tokens = tokenizer.encode(example)
print("Токены:", tokens)
print("Обратное преобразование:", tokenizer.decode(tokens))
print()

example = "17*456=7752"
tokens = tokenizer.encode(example)
print("Токены:", tokens)
print("Обратное преобразование:", tokenizer.decode(tokens))
print()

example = "1025&2=1"
tokens = tokenizer.encode(example)
print("Токены:", tokens)
print("Обратное преобразование:", tokenizer.decode(tokens))
