# Multi-Class Classification with a From-Scratch C# Neural Network (Iris)

A 3-layer Multi-Layer Perceptron (input -> hidden -> output, sigmoid activations, online backpropagation) written
in C#, applied to the classic **Iris** flower dataset. The network engine is the one provided in the previous
activity (`Engine/`). The Windows Forms window, data pipeline, training loop, tuning and evaluation are new.

## How to run

Requires Windows, Visual Studio 2022 (".NET desktop development" workload) and the .NET 8 SDK.

1. Open `BackpropActivity.csproj` in Visual Studio (File -> Open -> Project/Solution).
2. Press **F5** (or Ctrl+F5). The window opens and loads `data/iris.csv` automatically.
3. Use the window:
   - **Hidden neurons / Learning rates / Epochs:** the settings used for training.
   - **Train (new network):** builds a brand-new network with new random weights and trains it from epoch 1.
     The previous network, chart and log are discarded each time the button is pressed.
   - **Stop:** stops the training early.
   - **Find best settings (sweep):** tries 27 combinations, shows the top 10 in the log, and fills the best
     one into the boxes.
   - **Browse / Load:** load a different CSV (header row, numeric columns, last column = class name).
   - **Predict a row:** after training, type the measurements and press **Predict** to get the predicted class
     and the score of every class. **Random test row** takes a row from the held-out test set, fills the boxes,
     predicts it, and shows the actual class so you can see if the network was right.

The window shows the loss curve live (blue = training MSE, orange = validation MSE) and prints the training
log (MSE and accuracy per epoch), the confusion matrix, the metrics, and a majority-class baseline in the text box.

Output files are written to the `results/` folder next to the `.csproj`. They are overwritten by each new run:

| File | Contents |
|---|---|
| `training_log.csv` | Train and validation MSE and accuracy at epoch 1 and during training (the loss curve data) |
| `sweep_results.csv` | Every hyperparameter combination tried, ranked by validation accuracy |
| `evaluation_report.txt` | Accuracy, MSE, confusion matrix, per-class precision/recall/F1 on train and test sets, and the baseline |

## Project structure

| File | Responsibility |
|---|---|
| `Engine/InputN.cs`, `HiddenN.cs`, `OuputN.cs`, `BackNeural.cs` | Given neural network classes (`INeuron`, `HNeuron`, `ONeuron`, `NeuralNet`) |
| `Program.cs`, `MainForm.cs` | Start-up and the Windows Forms window |
| `LossChart.cs` | Custom control that draws the loss curve |
| `PreparedData.cs` | Loads, splits and scales the data in one step |
| `DataLoader.cs` | CSV parsing, skipping rows with missing/invalid values, one-hot targets |
| `Normalizer.cs` | Min-max scaling to [0,1], fitted on training data only |
| `DataSplitter.cs` | Stratified 70/15/15 train/validation/test split, shuffle helper |
| `Trainer.cs` | Builds the network, runs the epoch loop, logs the loss |
| `HyperparameterSweep.cs` | Grid search over hidden size, learning rates and epochs |
| `Evaluator.cs` | MSE, accuracy, confusion matrix, precision, recall, F1 |
| `ProjectPaths.cs` | Finds the results folder and the default dataset |

**Only change to the given engine:** in `NeuralNet`, `LRPOUT` and `LRPIN` were `const`; they are now public
fields with the same default values so the window and the sweep can set them.

## Data description

- **Dataset:** Iris (R. A. Fisher, 1936; available from the UCI Machine Learning Repository).
- **Size:** 150 rows, perfectly balanced: 50 *Iris-setosa*, 50 *Iris-versicolor*, 50 *Iris-virginica*.
- **Features (4, all in cm):** sepal length, sepal width, petal length, petal width.
- **Task:** multi-class classification, predicting the species from the four measurements.
- **Difficulty:** setosa is linearly separable from the other two; versicolor and virginica overlap slightly,
  so most errors occur between those two classes.
- **Other datasets included in `data/`:** `wine.csv` (178 rows, 13 features, 3 classes) and
  `breast_cancer.csv` (569 rows, 30 features, 2 classes: malignant / benign). Use **Browse** to load them; no
  code change is needed. Any CSV in the same format also works.
- **Missing values:** none in this file, but the loader skips (and reports) any row with an empty or
  non-numeric field.

## Method

1. **Preprocessing.** Rows are split with a stratified 70/15/15 split (105 train / 24 validation / 21 test)
   so each part keeps the class balance. Min-max scaling `x' = (x - min) / (max - min)` uses the minimum and
   maximum of the **training** rows only, then is applied to validation and test rows (values outside the
   training range are clamped to [0,1]). Scaling matters because the hidden layer uses sigmoid units, which
   saturate on large raw inputs.
2. **Network.** 4 inputs -> *H* hidden sigmoid neurons -> 3 sigmoid outputs, one per species. Targets are
   one-hot (setosa = `1,0,0`). The predicted class is the output neuron with the largest activation.
3. **Training.** Online backpropagation: for each sample, forward pass, output error
   `delta_out = o(1-o)(t-o)`, hidden error `delta_hid = h(1-h) * sum(delta_out * w)`, then weight and bias
   updates scaled by the learning rates. The training set is reshuffled every epoch.
4. **Loss.** Mean squared error averaged over the 3 output neurons and all training samples. The training
   accuracy in the log is counted during the epoch, from the same forward passes used for the loss.
5. **Tuning.** Grid search: hidden size {4, 8, 16} x (LRPOUT, LRPIN) {(0.05, 0.05), (0.20, 0.15), (0.50, 0.50)}
   x epochs {100, 300, 1000} = 27 settings. The engine's initial weights are random, so each setting is trained
   3 times and the results averaged. Settings are compared on the **validation** set; the test set is only used
   to report the final result of each Train run.

## Results

> Fill these in after running the program (copy from the window / `results/` files, and add a screenshot).

### Training log (loss across epochs)

```
(paste the "Epoch | TrainMSE | ValMSE | TrainAcc | ValAcc" lines here)
```

### Hyperparameter experiments

(paste the top rows of `results/sweep_results.csv` as a table)

| Hidden | LRPOUT | LRPIN | Epochs | Avg val. accuracy | Avg val. MSE |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

**Chosen configuration and why:** (write 3-5 sentences: which setting won; what happened with too few
epochs, too large a learning rate, too few hidden neurons; whether more hidden neurons or epochs stopped
helping.)

### Test-set evaluation

(paste the confusion matrix, the accuracy / precision / recall / F1 table, and the baseline line from
`results/evaluation_report.txt`)

**Analysis:** (write 3-5 sentences: overall accuracy compared with the majority-class baseline (for balanced
Iris the baseline is about 33%), which classes were confused (expect versicolor <->
virginica), the gap between training and test accuracy, and the limitation that the test set has only 21
samples, so one mistake moves accuracy by about 4.8 percentage points.)

## Limitations

- The dataset is small, so validation and test scores are noisy.
- The engine has no random-seed control for the initial weights, so results vary slightly between runs.
- The engine's `countgood()` stopping test compares floating-point values for exact equality, so training
  stops after a fixed number of epochs instead.
