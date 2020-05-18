from database import Model
import mongoengine
mongoengine.connect("network_training")


def print_models(models):
    for idx, model in enumerate(models):
        print("Number: {0:03} | LR: {1} | EP: {2} | NW: {3:03} | PND: {4:.3f} | PD: {5:.3f} | ACC: {6:.3f} | MAC: {7:.3f} | WEI: {8:.3f} | TAG: {9}".format(
            idx + 1,
            model.learning_rate,
            model.epochs,
            model.network_width,
            model.precision_no_delay,
            model.precision_delay,
            model.accuracy,
            model.macro_avg,
            model.weighted_avg,
            model.tag
        ))
    print("Averages:")
    print("PND: {0:.3f} | PD: {1:.3f} | ACC: {2:.3f} | MAC: {3:.3f} | WEI: {4:.3f}".format(
        models.average("precision_no_delay"),
        models.average("precision_delay"),
        models.average("accuracy"),
        models.average("macro_avg"),
        models.average("weighted_avg")
    ))

def filter_models(args_dict):
    wargs = {}
    if args_dict["lr"] != None:
        wargs["learning_rate"] = args_dict["lr"]
    if args_dict["e"] != None:
        wargs["epochs"] = args_dict["e"]
    if args_dict["nw"] != None:
        wargs["network_width"] = args_dict["nw"]
    if args_dict["t"] != None:
        wargs["tag"] = args_dict["t"]
    if args_dict["t"] == "None":
        wargs["tag"] = None
    return Model.objects(**wargs).order_by('-precision_delay')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View the saved models and their stats")
    parser.add_argument("-t", metavar="TAG", help="Filter the models with the tag")
    parser.add_argument('-lr', help='Filter with Learning rate', metavar="LEARNING_RATE")
    parser.add_argument('-e', help='Filter with Number of epochs', metavar="EPOCHS")
    parser.add_argument('-nw', help='Filter with Network Width', metavar="NETWORK_WIDTH")
    args = parser.parse_args()
    args = vars(args)
    print_models(filter_models(args))