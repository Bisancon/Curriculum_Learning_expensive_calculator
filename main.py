import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from DecoderTrans import TransformerDecoder
from tokenizer import CurriculumTokenizer
from torch.nn.utils.rnn import pad_sequence
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Загрузка конфигурации
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# Определение классов для работы с данными
class ExpressionDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = CurriculumTokenizer()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        expression = row['expression']
        result = row['result']
        # Преобразуем метку, вычитая минимальное значение метки
        result -= self.df['result'].min()  # Это приводит все метки к диапазону от 0 до N-1
        encoded_expression = self.tokenizer.encode(expression)
        return encoded_expression, result


def collate_batch(batch):
    expressions, results = zip(*batch)
    # Преобразуем каждое выражение в тензор
    expressions = [torch.tensor(expr) for expr in expressions]
    # Получаем длину каждого выражения
    lengths = [len(expr) for expr in expressions]
    # Паддим выражения
    padded_expressions = pad_sequence(expressions, batch_first=True, padding_value=0)
    # Преобразуем результаты в тензор
    results_tensor = torch.LongTensor(results)  # Тензор меток
    return padded_expressions, lengths, results_tensor


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 3:
            inputs, _, targets = batch
        elif len(batch) == 2:
            inputs, targets = batch
        else:
            raise ValueError("Unexpected batch format.")

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)[:, -1, :]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        # Логирование текущего батча
        writer.add_scalar("Train/Loss", loss.item(), epoch * len(loader) + batch_idx)

    accuracy = 100 * correct / total
    epoch_loss = total_loss / len(loader)
    writer.add_scalar("Train/Accuracy", accuracy, epoch)
    writer.add_scalar("Train/EpochLoss", epoch_loss, epoch)
    return epoch_loss, accuracy


def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                inputs, _, targets = batch
            elif len(batch) == 2:
                inputs, targets = batch
            else:
                raise ValueError("Unexpected batch format.")

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)[:, -1, :]
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            # Логирование текущего батча
            writer.add_scalar("Val/Loss", loss.item(), epoch * len(loader) + batch_idx)

    accuracy = 100 * correct / total
    epoch_loss = total_loss / len(loader)
    writer.add_scalar("Val/Accuracy", accuracy, epoch)
    writer.add_scalar("Val/EpochLoss", epoch_loss, epoch)
    return epoch_loss, accuracy


# Логирование графиков после завершения эксперимента
def log_final_plots(epochs, random_losses, curriculum_losses, random_accs, curriculum_accs):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(range(1, epochs + 1), random_losses, label='Random Loss')
    ax[0].plot(range(1, epochs + 1), curriculum_losses, label='Curriculum Loss')
    ax[0].set_title('Loss Comparison')
    ax[0].legend()

    ax[1].plot(range(1, epochs + 1), random_accs, label='Random Accuracy')
    ax[1].plot(range(1, epochs + 1), curriculum_accs, label='Curriculum Accuracy')
    ax[1].set_title('Accuracy Comparison')
    ax[1].legend()

    # Сохраняем графики в TensorBoard
    writer.add_figure("Results/Loss_Accuracy", fig)
    plt.close(fig)


# Случайное сэмплирование
def train_random_sampling(model, train_loader, val_loader, optimizer, criterion, device, epochs):
    random_train_losses, random_val_losses = [], []
    random_train_accs, random_val_accs = [], []

    for epoch in range(epochs):
        print(f"\nRandom Sampling Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch)
        random_train_losses.append(train_loss)
        random_train_accs.append(train_acc)
        random_val_losses.append(val_loss)
        random_val_accs.append(val_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    return random_train_losses, random_val_losses, random_train_accs, random_val_accs

# Curriculum learning
def train_curriculum_learning(model, train_dataset, val_loader, optimizer, criterion, device, epochs):
    # Создаем DataLoader без перемешивания
    curriculum_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

    curriculum_train_losses, curriculum_val_losses = [], []
    curriculum_train_accs, curriculum_val_accs = [], []

    for epoch in range(epochs):
        print(f"\nCurriculum Learning Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(model, curriculum_train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch)
        curriculum_train_losses.append(train_loss)
        curriculum_train_accs.append(train_acc)
        curriculum_val_losses.append(val_loss)
        curriculum_val_accs.append(val_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    return curriculum_train_losses, curriculum_val_losses, curriculum_train_accs, curriculum_val_accs

# Директория для логов TensorBoard
log_dir = config["logging"]["log_dir"]  # Укажите путь в вашем config.yaml
writer = SummaryWriter(log_dir)

# Установка устройства
device = torch.device(config["training"]["device"])

# Параметры данных
train_split = config["data"]["train_split"]
val_split = config["data"]["val_split"]
random_seed = config["data"]["random_seed"]
dataset_path = config["data"]["dataset_path"]

# Загрузка данных
df = pd.read_csv(dataset_path)

# Очистка выражений и удаление пробелов
def clean_expression(expression):
    # Удаляем пробелы
    expression = expression.replace(" ", "")
    return expression

df['expression'] = df['expression'].apply(clean_expression)

train_df, val_df = train_test_split(df, test_size=val_split, random_state=random_seed)

# Создание объектов Dataset
train_dataset = ExpressionDataset(train_df)
val_dataset = ExpressionDataset(val_df)

# DataLoader
batch_size = config["training"]["batch_size"]
shuffle = config["training"]["shuffle"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# Параметры модели
num_tokens = config["model"]["num_tokens"]
embedding_dim = config["model"]["embedding_dim"]
num_layers = config["model"]["num_layers"]
num_heads = config["model"]["num_heads"]
dropout = config["model"].get("dropout", 0.0)
num_classes = df['result'].max() - df['result'].min() + 1

model = TransformerDecoder(
    num_tokens=num_tokens,
    n_embd=embedding_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    num_classes=num_classes
).to(device)

# Параметры обучения
learning_rate = config["training"]["learning_rate"]
num_epochs = config["training"]["num_epochs"]

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Параметры обучения
epochs = num_epochs

# Запуск экспериментов
random_train_losses, random_val_losses, random_train_accs, random_val_accs = train_random_sampling(
    model, train_loader, val_loader, optimizer, criterion, device, epochs
)

# Функция для определения приоритета операции
def operation_priority(expression):
    if '+' in expression or '-' in expression:
        return 1  # Операции + и -
    elif '*' in expression or '^' in expression:
        return 2  # Операции * и ^
    return 3  # Если нет операций, можно отнести к самому низкому приоритету

# Сортировка датасета по сложности выражений
df['priority'] = df['expression'].apply(operation_priority)
df = df.sort_values(by='priority')
df = df.drop(columns=['priority'])

train_df, val_df = train_test_split(df, test_size=val_split, random_state=random_seed)
train_dataset = ExpressionDataset(train_df)
val_dataset = ExpressionDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

curriculum_train_losses, curriculum_val_losses, curriculum_train_accs, curriculum_val_accs = train_curriculum_learning(
    model, train_dataset, val_loader, optimizer, criterion, device, epochs
)

# Логирование итоговых графиков
log_final_plots(
    epochs,
    random_train_losses,
    curriculum_train_losses,
    random_train_accs,
    curriculum_train_accs
)

# Закрытие writer
writer.close()

# Визуализация потерь
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(range(1, epochs+1), random_train_losses, label='Random Train Loss')
plt.plot(range(1, epochs+1), random_val_losses, label='Random Val Loss')
plt.plot(range(1, epochs+1), curriculum_train_losses, label='Curriculum Train Loss')
plt.plot(range(1, epochs+1), curriculum_val_losses, label='Curriculum Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Comparison')

# Визуализация точности
plt.subplot(122)
plt.plot(range(1, epochs+1), random_train_accs, label='Random Train Accuracy')
plt.plot(range(1, epochs+1), random_val_accs, label='Random Val Accuracy')
plt.plot(range(1, epochs+1), curriculum_train_accs, label='Curriculum Train Accuracy')
plt.plot(range(1, epochs+1), curriculum_val_accs, label='Curriculum Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Comparison')

plt.show()
