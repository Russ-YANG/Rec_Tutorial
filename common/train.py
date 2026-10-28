"""
神经网络模型共用的训练循环。

每个模型只需要提供 compute_loss(model, batch) -> loss，
其余的优化器、epoch 循环、测试集评估都在这里。
"""
import torch
import torch.optim as optim


def get_device(device=None):
    if device is not None:
        return device
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(model, dataloader, compute_loss, device, optimizer=None):
    """跑一个 epoch，返回平均损失。传入 optimizer 时训练，否则只评估。"""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0

    with torch.set_grad_enabled(training):
        for batch in dataloader:
            batch = move_batch(batch, device)
            loss = compute_loss(model, batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = next(iter(batch.values())).size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / total_count


def fit(model, train_loader, test_loader, compute_loss, device,
        n_epochs=20, learning_rate=0.001, weight_decay=0.0, log_every=5, optimizer=None):
    """训练 n_epochs 轮，返回每轮的训练损失和测试损失。不传 optimizer 时用 Adam"""
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = {'train_loss': [], 'test_loss': []}

    for epoch in range(n_epochs):
        train_loss = run_epoch(model, train_loader, compute_loss, device, optimizer)
        test_loss = run_epoch(model, test_loader, compute_loss, device)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        if log_every and ((epoch + 1) % log_every == 0 or epoch == 0):
            print(f"Epoch [{epoch+1}/{n_epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}")

    return history
