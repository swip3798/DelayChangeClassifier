import argparse
import json
import train_network
import test_network
import mongoengine
from database import Model
from simple_bencher import Bencher

bench = Bencher("automated_network_training")

parser = argparse.ArgumentParser(description="Reads parameter json, trains all combinations which are possible and writes the results into a mongodb")

parser.add_argument("-f", metavar="FILENAME", help="Filename of the parameter.json", required=True)
parser.add_argument("-t", metavar="TAG", help="Tag to identify the training series", required=True)
args = parser.parse_args()
args = vars(args)

with open(args["f"], "r") as f:
    parameters = json.load(f)

parameter_combinations = []

for lr in parameters["lr"]:
    for nw in parameters["nw"]:
        for ep in parameters["ep"]:
            parameter_combinations.append([lr, ep, nw])

print("{} parameter combinations, training started...".format(len(parameter_combinations)))
bench.start()
mongoengine.connect("network_training")
for parameter in parameter_combinations:
    runtime, model_path = train_network.train_network(parameter[0], parameter[1], "cpu", parameter[2])
    classification_report = test_network.test_network(parameter[2], model_path)
    model = Model.save_model(runtime, "cpu", parameter[0], parameter[1], parameter[2], model_path, classification_report, tag = args["t"])
    model.save()
bench.stop()
print("Trained all outcomes [{}]".format(bench.get_time()))