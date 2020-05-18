import torch
from binary_classify import BinaryClassification, test_loader, label_test
from sklearn.metrics import confusion_matrix, classification_report
from binary_classify.constants import PATH, NETWORK_WIDTH
import tqdm



def test_network(network_width, path = PATH):
    model = BinaryClassification(network_width)
    model.load_state_dict(torch.load(path))
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    confusion_matrix(label_test, y_pred_list)
    return classification_report(label_test, y_pred_list, output_dict=True)

if __name__ == "__main__":
    import json
    import argparse
    parser = argparse.ArgumentParser(description="Test a models performance")
    parser.add_argument("-nw", metavar="NETWORK_WIDTH", help="Width of the network", default=NETWORK_WIDTH)
    parser.add_argument("-f", metavar="FILENAME", help="Filename of the model state", default=PATH)
    args = parser.parse_args()
    args = vars(args)
    print(json.dumps(test_network(int(args["nw"]), args["f"]), indent=2))