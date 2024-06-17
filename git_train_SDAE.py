import os
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
import torch
from torch import nn
from sklearn import preprocessing
from pytorchtools import EarlyStopping
from tqdm import trange
import numpy as np
from fastdtw import fastdtw
import DAE_git
seed_value = 5678          #5678
# 무작위 시드를 설정합니다.
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)

ex_list = []

DATA_SHAPE=30000
EPOCHS = 100000
BATCH_SIZE = 4    #4
FOLDS=1
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=30000, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch size, 1, 30000]
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        # LSTM layer
        lstm_out, _ = self.lstm(x, hidden)

        # Flatten the output
        lstm_out = lstm_out[:, -1, :]

        # Fully connected layer
        out = self.fc(lstm_out)

        return lstm_out,out

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)
        return (h0, c0)
results = {}

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

for i in range(1):

    ############################
    # data load
    ############################

    # temp = np.load('/...')
    # temp_noised = np.load('/...')
    # temp_noised_test = np.load('/...')

    x = temp[:, :-1]
    y = temp[:, -1].reshape(-1, 1)
    x_len = len(x)
    x_noised = temp_noised[:, :-1]
    y_noised = temp_noised[:, -1]
    x_noised_test = temp_noised_test[:, :-1]
    y_noised_test = temp_noised_test[:, -1]

    min_max_scaler = preprocessing.MinMaxScaler()
    # min_max_scaler = preprocessing.StandardScaler()
    x = min_max_scaler.fit_transform(x)
    x = x.reshape(x.shape[0], 1, x.shape[1])
    x = torch.from_numpy(x.astype(np.float32))
    y = torch.from_numpy(y).type(torch.LongTensor)
    dataset = torch.utils.data.TensorDataset(x, y)
    x = x.reshape(x_len, -1)

    origin_data = np.hstack([x, y])

    origin_label_dict = {}

    for data_with_label in origin_data:
        data = data_with_label[:-1]
        label = data_with_label[-1]
        if label not in origin_label_dict:
            origin_label_dict[label] = []
        origin_label_dict[label].append(data)

    # print(len(origin_label_dict[1.0]))

    x_noised = min_max_scaler.transform(x_noised)
    # x_noised = add_origin_noise(x_noised, a)
    x_noised = x_noised.reshape(x_noised.shape[0], 1, x_noised.shape[1])
    x_noised = torch.from_numpy(x_noised.astype(np.float32))
    y_noised = torch.from_numpy(y_noised).type(torch.LongTensor)
    noised_dataset = torch.utils.data.TensorDataset(x_noised, y_noised)

    x_noised_test = min_max_scaler.transform(x_noised_test)
    # x_noised_test = add_origin_noise(x_noised_test,a)

    x_noised_test = x_noised_test.reshape(x_noised_test.shape[0], 1, x_noised_test.shape[1])
    x_noised_test = torch.from_numpy(x_noised_test.astype(np.float32))
    y_noised_test = torch.from_numpy(y_noised_test).type(torch.LongTensor)
    noised_dataset_test = torch.utils.data.TensorDataset(x_noised_test, y_noised_test)

    # load clf
    torch.cuda.empty_cache()
    lr_lstm, bs_lstm, hidden_size, num_layers = [1e-05, 8, 256, 4]
    model1 = LSTMModel(hidden_size, num_layers, 5)
    model1.load_state_dict(torch.load('./temp_mean.pt'))
    model1.to(DEVICE)
    model1.eval()

    autoencoder = DAE_git.Autoencoder_instance(DATA_SHAPE).to(DEVICE)

    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=0.00001)
    # optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    criterion = nn.MSELoss()

    autoencoder.apply(reset_weights)
    autoencoder.train()
    avg_loss = 0
    tepoch = trange(EPOCHS, desc="Epochs")
    early_stopping = EarlyStopping(patience=20, verbose=False, path=f'./DTW_DAE.pt')

    # print(f'FOLD {fold}')
    print('--------------------------------')
    dataset_size = len(noised_dataset)
    train_size = int(0.8 * dataset_size)  # 80%를 train set으로 사용
    test_size = dataset_size - train_size  # 나머지 20%를 test set으로 사용
    from torch.utils.data import random_split

    # 데이터셋 분할
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE, shuffle=True)
    valloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_dataset.__len__())
    R_testloader = torch.utils.data.DataLoader(
                                  noised_dataset_test,
                                  batch_size=noised_dataset_test.__len__())
    for epoch in tepoch:
        # if epoch == 200:
        #     print("change lr 1/10")
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.000001
        # if epoch == 80:
        #     print("change lr 1/10")
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.000001
        autoencoder.train()
        for step, (x, label) in enumerate(trainloader):

            # print(x.shape)
            # bs = x.size(0)
            # noisy_x = add_noise_wave(x,noise=noise_arr, batch_size=bs)
            # noisy_x = add_noise(noisy_x,DATA_SHAPE,1)
            noisy_x = x.view(-1, DATA_SHAPE).to(DEVICE)
            # noisy_x = x.view(-1, 1,DATA_SHAPE).to(DEVICE)
            # y = x.view(-1, DATA_SHAPE).to(DEVICE)
            target = []
            for data_idx, data_class in enumerate(label):
                # print('@@@@',data_idx, data_class)
                # print('label:', float(i))
                # print('label_index:',random.randint(0,len(origin_label_dict[float(i)])-1))

                # target.append(origin_label_dict[float(i)][random.randint(0,len(origin_label_dict[float(i)])-1)])
                low = 10
                low_idx = 0

                for idx in range(len(origin_label_dict[float(data_class)])):
                    # print("1",x[data_idx].shape)
                    # print('2',origin_label_dict[float(data_class)][idx].shape)
                    mmd_temp, path = fastdtw(x[data_idx].reshape(1,-1), origin_label_dict[float(data_class)][idx].reshape(1,-1))
                    if low>mmd_temp:
                        low = mmd_temp
                        low_idx = idx
                # print("@@@@@@@@@@@")
                # print('low_mmd:',low)
                # print('low_idx:',low_idx)
                # print('data_class:',data_class)
                # print("@@@@@@@@@@@")
                target.append(origin_label_dict[float(data_class)][low_idx])

            # print(len(target))
            target = torch.tensor(np.array(target)).to(torch.float32).to(DEVICE)
            # target = torch.cat(target)
            # print(target.shape)
            # print(noisy_x.shape)
            # print(noisy_x.shape)
            encoded, decoded = autoencoder(noisy_x)

            loss = criterion(decoded, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tepoch.set_postfix(loss=loss.item())
        # loss_list.append(loss.item())

        final_val_loss = 0
        autoencoder.eval()
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, (x, label) in enumerate(valloader):
                # noisy_x = add_noise(x, DATA_SHAPE, 0)
                # bs = x.size(0)
                # noisy_x = add_noise_wave(x,noise_arr,bs)
                # noisy_x = add_noise(noisy_x, DATA_SHAPE, 1)
                noisy_x = x.view(-1, DATA_SHAPE).to(DEVICE)
                # noisy_x = x.view(-1, 1, DATA_SHAPE).to(DEVICE)

                # y = x.view(-1, DATA_SHAPE).to(DEVICE)
                target = []
                for data_idx, data_class in enumerate(label):
                    # print('label:', float(i))
                    # print('label_index:',random.randint(0,len(origin_label_dict[float(i)])-1))

                    # target.append(origin_label_dict[float(i)][random.randint(0,len(origin_label_dict[float(i)])-1)])
                    low = 10
                    low_idx = 0

                    for idx in range(len(origin_label_dict[float(data_class)])):
                        # print("1",x[data_idx].shape)
                        # print('2',origin_label_dict[float(data_class)][idx].shape)
                        mmd_temp, path = fastdtw(x[data_idx].reshape(1, -1),
                                               origin_label_dict[float(data_class)][idx].reshape(1, -1))
                        if low > mmd_temp:
                            low = mmd_temp
                            low_idx = idx
                    # print("@@@@@@@@@@@")
                    # print('low_mmd:',low)
                    # print('low_idx:',low_idx)
                    # print('data_class:',data_class)
                    # print("@@@@@@@@@@@")
                    target.append(origin_label_dict[float(data_class)][low_idx])
                # print(len(target))
                target = torch.tensor(np.array(target)).to(torch.float32).to(DEVICE)
                # print(target.shape, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!', target, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                # print('eval_target.shape:',target.shape)
                # print(noisy_x.shape)
                # label = label.to(DEVICE)
                encoded, decoded = autoencoder(noisy_x)

                val_loss = criterion(decoded, target)
                final_val_loss = val_loss.item()

            early_stopping(val_loss, autoencoder)
            if early_stopping.early_stop:

                correct, total = 0, 0
                print("R_dataset test")
                autoencoder.load_state_dict(torch.load('./DTW_DAE.pt'))

                with torch.no_grad():
                  # Iterate over the test data and generate predictions
                    for i, (inputs, targets) in enumerate(R_testloader):
                        # print("before:", inputs.shape)
                        # print(torch.std(inputs, dim=(2), unbiased=False))
                        # inputs = add_noise(inputs, 30000, 0)
                        # inputs = add_noise_wave(inputs, 30000, 0)

                        # inputs = inputs.view(-1,1, 30000).to(device)
                        # inputs = inputs.view(-1, 30000).to(device)
                        # print("after:", inputs.shape)
                        #
                        # targets = targets.to(device)
                        # plt.plot(inputs[20][0].cpu())
                        # plt.title(f"before_{targets[20]}")
                        # plt.show()
                        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                        with torch.no_grad():
                            encoded, decoded = autoencoder(inputs)
                        decoded = decoded.reshape(-1,1,30000)
                        # print("decoded:", decoded.shape)
                        # plt.plot(decoded[20][0].cpu())
                        # plt.title(f"after{targets[20]}")
                        # plt.show()
                        lstm_out, outputs = model1(decoded)
                        # lstm_out, outputs = model1(inputs)
                        _, predicted = torch.max(outputs, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                        print('\ntest_Actual   :', targets)
                        print('test_Predicted:', predicted)
                        print('total:', total, ' Correct:', correct)
                    # Print accuracy
                    print('Accuracy: %d %%' % ( 100.0 * correct / total))
                    print('--------------------------------')

                result_lstm = ( 100.0 * correct / total)
                ex_list.append(result_lstm)
                print('ex_list:', ex_list)

                break
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Allocated memory_Before: {allocated_memory / 1024 ** 2:.2f} MB")
    del model1
    del autoencoder
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Allocated memory_After: {allocated_memory / 1024 ** 2:.2f} MB")

print("RESULT")
sum = 0
# for key, value in results.items():
#     print(f'Fold {key}: {value} %')
#     sum += value
#
# print(f'Average: {sum/len(results.items())} %')
