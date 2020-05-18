from mongoengine import Document, StringField, IntField, FloatField, BinaryField, DateTimeField
import datetime

class Model(Document):
    timestamp = DateTimeField()
    runtime = FloatField()
    device_type = StringField()
    learning_rate = FloatField()
    epochs = IntField()
    network_width = IntField()
    precision_no_delay = FloatField()
    precision_delay = FloatField()
    accuracy = FloatField()
    macro_avg = FloatField()
    weighted_avg = FloatField()
    model_file = BinaryField()
    tag = StringField(default = "")

    @staticmethod
    def save_model(runtime, device_type, learning_rate, epochs, network_width, path, classification_report, tag = ""):
        precision_no_delay = classification_report["0"]["precision"]
        precision_delay = classification_report["1"]["precision"]
        accuracy = classification_report["accuracy"]
        macro_avg = classification_report["macro avg"]["precision"]
        weighted_avg = classification_report["weighted avg"]["precision"]
        with open(path, "rb") as f:
            model_bin = f.read()
        return Model(
            timestamp = datetime.datetime.now(), 
            runtime = runtime,
            device_type = device_type, 
            learning_rate = learning_rate, 
            epochs = epochs, 
            network_width = network_width,
            precision_delay = precision_delay,
            precision_no_delay = precision_no_delay,
            accuracy = accuracy,
            macro_avg = macro_avg,
            weighted_avg = weighted_avg,
            model_file = model_bin,
            tag = tag
        )
