# DelayClassifier

Auf Basis dieses [Artikels](https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89)

## Vorraussetzung
Für das Neurale Netzwerk wird PyTorch benötigt. Die Installation wird auf der [Startseite](https://pytorch.org/) von PyTorch beschrieben.   
Alle übrigen notwendigen Python Module befinden sich in der `requirements.txt` und können mit folgendem Befehl installiert werden.
```
pip install -r requirements.txt
```
Darüber hinaus werden Trainingsdaten benötigt. Diese werden im CSV-Format eingelesen und sind wie die .csv.example Dateien strukturiert.
## Benutzung

Zum Training eines bestimmten Models
```
python train_network.py [-h] -lr LEARNING_RATE -e EPOCHS [-d DEVICE_TYPE] -nw NETWORK_WIDTH

Train the network with given parameters

optional arguments:
  -h, --help         show this help message and exit
  -lr LEARNING_RATE  Learning rate
  -e EPOCHS          Number of epochs
  -d DEVICE_TYPE     Device type
  -nw NETWORK_WIDTH  Width of the network
```   
   
Zum Testen eines Netzwerks   
```
python test_network.py [-h] [-nw NETWORK_WIDTH] [-f FILENAME]

Test a models performance

optional arguments:
  -h, --help         show this help message and exit
  -nw NETWORK_WIDTH  Width of the network
  -f FILENAME        Filename of the model state
```

## Automatisiertes Training
Das automatisierte Training benötigt eine lokal laufende MongoDB Instanz.  
     
Alle zu testende Parameter können anschließend in einer JSON-Datei festgelegt werden. Alle möglichen Kombinationen werden anschließend trainiert, getestet und die Test-Ergebnisse sowie die State-Datei des Modells in der MongoDB `network_training` abgespeichert. Darüber hinaus kann ein `tag` angegeben werden, um besser in den getesteten Netzwerken zu filtern.

```javascript
{
    "lr": [0.001, 0.002, 0.003, 0.004, 0.005],
    "ep": [30, 50, 70, 100],
    "nw": [32, 48, 64, 96, 128]
}
```

```
python automated_training.py [-h] -f FILENAME -t TAG

Reads parameter json, trains all combinations which are possible and writes the results into a mongodb

optional arguments:
  -h, --help   show this help message and exit
  -f FILENAME  Filename of the parameter.json
  -t TAG       Tag to identify the training series
python .\automated_training.py -f parameters.json -t Training1
100 parameter combinations, training started...
```