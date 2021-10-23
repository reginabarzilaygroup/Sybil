# Sybil

Lung Cancer Risk Prediction

## Run the model

You can load our pretrained model trained on the NLST dataset, and score a given DICOM serie as follows:

```python
from sybil import Serie, Sybil

# Create a Serie from a list of DICOM paths
serie = Serie([dicom_path_1, dicom_path_2, ...])

# Load a trained model
model = Sybil.from_pretrained("sybil_large")

# Get risk scores
scores = model.score(serie)
```


## Evaluate

If you wish to evaluate our model on a 

```python
from sybil import Serie, Sybil

# Create a Serie from a list of DICOM paths
series: List[Serie] = [...]

# Train a new model
# Load a trained model
model = Sybil.load("sybil_large")
model.evaluate(series)
```


## Training


```python
from sybil import Serie, Sybil

# Create a Serie from a list of DICOM paths
series: List[Serie] = [...]

# Train a new model
model = Sybil()
model.fit(series)
```

See our [documentation] for a full description of Sybil's training parameters.
## Finetuning

```python
from sybil import Serie, Sybil

# Create a Serie from a list of DICOM paths
series: List[Serie] = [...]

# Train a new model
model = Sybil()
model.finetune(series)
```

See our [documentation] for a full description of Sybil's training parameters.