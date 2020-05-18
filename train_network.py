from binary_classify import BinaryClassification, train_loader, test_loader, get_preprocessed_dataframe

from binary_classify.constants import PATH, TRAIN_BATCH_SIZE

import torch
import torch.nn as nn
import torch.optim as optim

from simple_bencher import Bencher

import tqdm

def train_network(learning_rate, epochs, device_type, network_width):
    print("Data from csv file loaded")
    device = torch.device(device_type)
    model = BinaryClassification(network_width=network_width)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def binary_acc(y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum/y_test.shape[0]
        acc = torch.round(acc * 100)
        
        return acc

    print("Start training")
    print("Learning rate:", learning_rate)
    print("Epochs:", epochs)
    print("Network:", model)
    print("Using device:", device)
    bench = Bencher("binary_network_training")
    bench.start()
    model.train()
    total_train_data = len(train_loader) * TRAIN_BATCH_SIZE
    pbar = tqdm.tqdm(total = epochs * total_train_data)
    for e in range(1, epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            y_pred = model(X_batch)
            
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            pbar.update(len(y_batch))
            
        
        pbar.set_postfix(loss="{:.5}".format(epoch_loss/len(train_loader)), acc=epoch_acc/len(train_loader))
        # print('Epoch {0:03d}: | Loss: {1:.5f} | Acc: {2:.5f}'.format(e, epoch_loss/len(train_loader), epoch_acc/len(train_loader)))
    pbar.close()
    torch.save(model.state_dict(), PATH)
    bench.stop()
    print("Training finished [{}s]".format(bench.get_time()))
    return bench.get_time(), PATH

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the network with given parameters")
    parser.add_argument('-lr', help='Learning rate', required=True, metavar="LEARNING_RATE", default="0.001")
    parser.add_argument('-e', help='Number of epochs', required=True, metavar="EPOCHS")
    parser.add_argument('-d', help='Device type', required=False, default='cpu', metavar="DEVICE_TYPE")
    parser.add_argument('-nw', help='Width of the network', required=True, metavar="NETWORK_WIDTH")

    args = parser.parse_args()
    args = vars(args)
    args["lr"] = float(args["lr"])
    args["e"] = int(args["e"])
    args["nw"] = int(args["nw"])
    train_network(args["lr"], args["e"], args["d"], args["nw"])