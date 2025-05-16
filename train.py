import torch


def train(model, device, train_loader, test_loader, criterion, optimizer, num_epoch):
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            total += target.size(0)

        train_accuracy = train_correct / total * 100

        # 在每个 epoch 中后调用测试模型返回的结果，以计算测试损失和测试准确率
        test_loss, test_accuracy = test(model, test_loader, device, criterion)

        torch.save(model.state_dict(), './model/sample_model-' + str(epoch) + '.pt')  # 保存模型

        print(f'Epoch: {epoch+1}/{num_epoch} | '
            f'Train Loss: {train_loss / len(train_loader):.5f} | '
            f'Train Accuracy: {train_accuracy:.2f}% | '
            f'Test Loss: {test_loss:.5f} | '
            f'Test Accuracy: {test_accuracy:.2f}%')


def test(model, data_loader, device, criterion):
    model.eval()
    test_loss = 0
    test_correct = 0
    total = 0

    # 测试函数无需梯度计算
    with torch.no_grad():
        for data, target in data_loader:
            # 将 batch 中的每一对样本数据都传到 GPU 设备上
            data, target = data.to(device), target.to(device)
            # 获得输出结果
            output = model(data)
            # 计算损失
            loss = criterion(output, target)
            # 损失累加
            test_loss += loss.item()
            # 在每一行找到最大概率的索引，这个索引即为模型的预测类别
            pred = output.argmax(dim=1)
            # 1. 将 pred 与 target 作比较
            # 2. 例如，pred=[3, 2, 5, 0]，target=[3, 2, 4, 0]
            # 3. 则 (pred == target) = [True, True, False, True]
            # 4. 那么就意味着有 3 个样本预测正确了，并累加预测正确的样本数量，即 3
            test_correct += (pred == target).sum().item()
            total += target.size(0)

    # 计算评价损失和平均准确率
    average_loss = test_loss / len(data_loader)
    average_accuracy = test_correct / total * 100
    return average_loss, average_accuracy


def predict(model, data_loader, device):
    model.eval()
    target = list()

    with torch.no_grad():
        for data in data_loader:
            # 将 batch 中的每一对样本数据都传到 GPU 设备上
            data = data[0].to(device)
            # 获得输出结果
            output = model(data)
            # 在每一行找到最大概率的索引，这个索引即为模型的预测类别
            pred = output.argmax(dim=1)
            # 例如，pred=[3, 2, 5, 0]
            target.extend(pred.cpu().numpy().tolist())

    return target
