from model import Transformer, TranslationDataset
from collections import Counter
# Дополнительные инпуты
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Токенизация
def tokenize(text):
    return text.lower().split()

# Функция для создания словаря токенов
def build_vocab(sentences):
    token_counter = Counter()
    for sentence in sentences:
        token_counter.update(tokenize(sentence))
    
    vocab = {word: idx for idx, (word, _) in enumerate(token_counter.items(), start=4)}
    vocab["<PAD>"] = 0  # Padding token
    vocab["<SOS>"] = 1  # Start of sentence token
    vocab["<EOS>"] = 2  # End of sentence token
    vocab["<UNK>"] = 3  # Unknown token (на случай неизвестных слов)
    
    return vocab

# Пример создания словарей для русского и английского языков
source_sentences = ["Привет мир", "Как дела"]
target_sentences = ["Hello world", "How are you"]

# Создание словарей для русского и английского языков
source_vocab = build_vocab(source_sentences)
target_vocab = build_vocab(target_sentences)


# Параметры модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_size = 256
num_layers = 6
heads = 8
forward_expansion = 4
dropout = 0.1
max_length = 100
learning_rate = 3e-4
batch_size = 32
num_epochs = 10

# Примерные размеры словарей
src_vocab_size = len(source_vocab)
trg_vocab_size = len(target_vocab)
src_pad_idx = source_vocab["<PAD>"]
trg_pad_idx = target_vocab["<PAD>"]

# Инициализация модели
model = Transformer(
    src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size,
    num_layers, forward_expansion, heads, dropout, max_length
).to(device)

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Тренировочный цикл
def train(model, dataset, optimizer, criterion, device, epochs=10):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for batch_idx, (src, trg) in enumerate(loader):
            # Преобразуем source и target из кортежа в строку
            src_sentence = src[0]  # Извлекаем строку из кортежа
            trg_sentence = trg[0]  # Извлекаем строку из кортежа
            
            # Токенизация предложений
            src_tokens = [source_vocab.get(token, source_vocab["<UNK>"]) for token in tokenize(src_sentence)]
            trg_input_tokens = [target_vocab.get(token, target_vocab["<UNK>"]) for token in tokenize(trg_sentence)]

            # Создаем тензоры и преобразуем их в целочисленный тип
            src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(device)
            trg_input_tensor = torch.tensor(trg_input_tokens, dtype=torch.long).unsqueeze(0).to(device)

            # Таргет для сравнения
            trg_output = trg_input_tensor[:, 1:].reshape(-1)

            # Модель прогнозирует перевод
            output = model(src_tensor, trg_input_tensor[:, :-1])
            output = output.reshape(-1, output.shape[2])

            # Вычисляем ошибку
            loss = criterion(output, trg_output)

            # Обновляем градиенты
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")


# Пример использования
train_dataset = TranslationDataset("datasets/source.txt", "datasets/target.txt")
train(model, train_dataset, optimizer, criterion, device, num_epochs)