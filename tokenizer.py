class CurriculumTokenizer:
    def __init__(self):
        self.unk_token = 0  # Для неизвестного токена
        self.vocab = {
            '<PAD>': 0,  # Для дополнения последовательности
            '<SOS>': 1,  # Начало последовательности
            '<EOS>': 2,  # Конец последовательности
            '<UNK>': 3,  # Для неизвестных токенов
            '+': 4,
            '-': 5,
            '*': 6,
            '^': 7,
            '=': 8,
            # Добавим другие токены
        }
        # Добавление цифр
        for i in range(10):
            self.vocab[str(i)] = 10 + i

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)  # Размер словаря

    def encode(self, expression):
        tokens = [self.vocab['<SOS>']]
        for char in expression:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab['<UNK>'])  # Неизвестный токен
        tokens.append(self.vocab['<EOS>'])
        return tokens

    def decode(self, tokens):
        return ''.join(self.reverse_vocab[token] for token in tokens)

    def tokenize(self, expression):
        tokens = [self.vocab['<SOS>']]  # Начинаем с токена начала последовательности
        for char in expression:
            if char in self.vocab:
                tokens.append(self.vocab[char])  # Преобразуем символ в токен
            else:
                tokens.append(self.vocab['<UNK>'])  # Токен для неизвестных символов
        tokens.append(self.vocab['<EOS>'])  # Конец последовательности
        return tokens

    def safe_tokenize(self, expression):
        tokens = self.tokenize(expression)
        return [token if token < self.vocab_size else self.unk_token for token in tokens]


# # Пример использования
# tokenizer = CurriculumTokenizer()
#
# example = "35-14=21"
# tokens = tokenizer.encode(example)
# print("Токены:", tokens)
# print("Обратное преобразование:", tokenizer.decode(tokens))
# print()
#
# example = "17*456=7752"
# tokens = tokenizer.encode(example)
# print("Токены:", tokens)
# print("Обратное преобразование:", tokenizer.decode(tokens))
# print()
#
# example = "1025&2=1"
# tokens = tokenizer.encode(example)
# print("Токены:", tokens)
# print("Обратное преобразование:", tokenizer.decode(tokens))
#
# example = "(1025&2)+3=1"
# tokens = tokenizer.encode(example)
# print("Токены:", tokens)
# print("Обратное преобразование:", tokenizer.decode(tokens))
