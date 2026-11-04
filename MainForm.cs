using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Backprop
{
    // the main window: load data, set the settings, train, and see the results
    public class MainForm : Form
    {
        // if the dataset has more columns than this, the Predict boxes are not shown
        private const int MaxTypedInputs = 8;

        // buttons, boxes and labels
        private TextBox txtDataPath;
        private Button btnBrowse;
        private Button btnLoad;
        private Button btnTrain;
        private Button btnStop;
        private Button btnSweep;
        private NumericUpDown numHidden;
        private NumericUpDown numLrOut;
        private NumericUpDown numLrIn;
        private NumericUpDown numEpochs;
        private Label lblDataInfo;
        private Label lblStatus;
        private ProgressBar progressBar;
        private TextBox txtLog;
        private LossChart chart;

        // the Predict panel
        private FlowLayoutPanel predictRow;
        private Button btnPredict;
        private Button btnRandomRow;
        private Label lblPrediction;
        private readonly List<TextBox> predictInputs = new List<TextBox>();
        private readonly List<Control> predictDynamic = new List<Control>();

        // the loaded data, the current network, the stop signal, and a busy flag
        private PreparedData data;
        private NeuralNet net;
        private CancellationTokenSource cts;
        private bool isBusy;
        private readonly Random random = new Random();

        public MainForm()
        {
            Text = "Neural Network Trainer";
            ClientSize = new Size(1000, 700);
            MinimumSize = new Size(820, 560);
            StartPosition = FormStartPosition.CenterScreen;

            BuildControls();
            LoadDefaultData();
        }

        // ---------- building the window ----------

        // creates all the controls and places them in the window
        private void BuildControls()
        {
            // main layout: one column with several rows
            TableLayoutPanel root = new TableLayoutPanel();
            root.Dock = DockStyle.Fill;
            root.ColumnCount = 1;
            root.RowCount = 8;
            root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
            Controls.Add(root);

            // row 1: dataset file
            txtDataPath = new TextBox();
            txtDataPath.Width = 460;
            btnBrowse = NewButton("Browse...", btnBrowse_Click);
            btnLoad = NewButton("Load", btnLoad_Click);
            FlowLayoutPanel dataRow = NewRow();
            dataRow.Controls.Add(NewLabel("Dataset (CSV):"));
            dataRow.Controls.Add(txtDataPath);
            dataRow.Controls.Add(btnBrowse);
            dataRow.Controls.Add(btnLoad);
            AddRow(root, dataRow, false);

            // row 2: settings
            numHidden = NewNumber(1, 512, 8, 1, 0, 60);
            numLrOut = NewNumber(0.01M, 5M, 0.20M, 0.01M, 2, 65);
            numLrIn = NewNumber(0.01M, 5M, 0.15M, 0.01M, 2, 65);
            numEpochs = NewNumber(1, 100000, 500, 100, 0, 75);
            FlowLayoutPanel settingsRow = NewRow();
            settingsRow.Controls.Add(NewLabel("Hidden neurons:"));
            settingsRow.Controls.Add(numHidden);
            settingsRow.Controls.Add(NewLabel("Learning rate (output):"));
            settingsRow.Controls.Add(numLrOut);
            settingsRow.Controls.Add(NewLabel("Learning rate (input):"));
            settingsRow.Controls.Add(numLrIn);
            settingsRow.Controls.Add(NewLabel("Epochs:"));
            settingsRow.Controls.Add(numEpochs);
            AddRow(root, settingsRow, false);

            // row 3: action buttons
            btnTrain = NewButton("Train (new network)", btnTrain_Click);
            btnStop = NewButton("Stop", btnStop_Click);
            btnStop.Enabled = false;
            btnSweep = NewButton("Find best settings (sweep)", btnSweep_Click);
            FlowLayoutPanel buttonRow = NewRow();
            buttonRow.Controls.Add(btnTrain);
            buttonRow.Controls.Add(btnStop);
            buttonRow.Controls.Add(btnSweep);
            AddRow(root, buttonRow, false);

            // row 4: information about the data
            lblDataInfo = new Label();
            lblDataInfo.AutoSize = true;
            lblDataInfo.Margin = new Padding(10, 4, 4, 4);
            AddRow(root, lblDataInfo, false);

            // row 5: chart on the left, text log on the right
            TableLayoutPanel center = new TableLayoutPanel();
            center.Dock = DockStyle.Fill;
            center.ColumnCount = 2;
            center.RowCount = 1;
            center.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
            center.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
            center.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));

            chart = new LossChart();
            chart.Dock = DockStyle.Fill;
            chart.Margin = new Padding(8, 4, 4, 4);
            center.Controls.Add(chart, 0, 0);

            txtLog = new TextBox();
            txtLog.Multiline = true;
            txtLog.ReadOnly = true;
            txtLog.ScrollBars = ScrollBars.Both;
            txtLog.WordWrap = false;
            txtLog.Font = new Font("Consolas", 9F);
            txtLog.BackColor = Color.White;
            txtLog.Dock = DockStyle.Fill;
            txtLog.Margin = new Padding(4, 4, 8, 4);
            center.Controls.Add(txtLog, 1, 0);
            AddRow(root, center, true);

            // row 6: Predict panel (the boxes are made in RebuildPredictRow)
            btnPredict = NewButton("Predict", btnPredict_Click);
            btnRandomRow = NewButton("Random test row", btnRandomRow_Click);
            lblPrediction = new Label();
            lblPrediction.AutoSize = true;
            lblPrediction.Margin = new Padding(10, 7, 2, 0);
            predictRow = NewRow();
            AddRow(root, predictRow, false);
            RebuildPredictRow();

            // row 7: status text
            lblStatus = new Label();
            lblStatus.Text = "Ready";
            lblStatus.AutoSize = true;
            lblStatus.Margin = new Padding(10, 4, 4, 4);
            AddRow(root, lblStatus, false);

            // row 8: progress bar
            progressBar = new ProgressBar();
            progressBar.Dock = DockStyle.Fill;
            progressBar.Margin = new Padding(10, 2, 10, 8);
            AddRow(root, progressBar, false);
        }

        // adds a control as the next row (fill = true makes the row take all the free space)
        private static void AddRow(TableLayoutPanel table, Control control, bool fill)
        {
            table.RowStyles.Add(fill ? new RowStyle(SizeType.Percent, 100F) : new RowStyle(SizeType.AutoSize));
            table.Controls.Add(control, 0, table.RowStyles.Count - 1);
        }

        // creates a row that places its controls side by side
        private static FlowLayoutPanel NewRow()
        {
            FlowLayoutPanel row = new FlowLayoutPanel();
            row.Dock = DockStyle.Fill;
            row.AutoSize = true;
            row.WrapContents = true;
            row.Padding = new Padding(6, 2, 6, 0);
            return row;
        }

        // creates a text label
        private static Label NewLabel(string text)
        {
            Label label = new Label();
            label.Text = text;
            label.AutoSize = true;
            label.Margin = new Padding(8, 7, 2, 0);
            return label;
        }

        // creates a button
        private static Button NewButton(string text, EventHandler onClick)
        {
            Button button = new Button();
            button.Text = text;
            button.AutoSize = true;
            button.MinimumSize = new Size(80, 28);
            button.Click += onClick;
            return button;
        }

        // creates a number box
        private static NumericUpDown NewNumber(decimal min, decimal max, decimal value, decimal step, int decimals, int width)
        {
            NumericUpDown box = new NumericUpDown();
            box.Minimum = min;
            box.Maximum = max;
            box.Increment = step;
            box.DecimalPlaces = decimals;
            box.Value = value;
            box.Width = width;
            return box;
        }

        // ---------- loading data ----------

        // loads data/iris.csv when the program starts
        private void LoadDefaultData()
        {
            string path = ProjectPaths.FindDefaultDataFile();
            if (path == null)
            {
                lblDataInfo.Text = "No dataset loaded. Click Browse and choose a CSV file.";
                return;
            }
            txtDataPath.Text = path;
            LoadData(path);
        }

        // opens a file dialog and loads the chosen file
        private void btnBrowse_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog dialog = new OpenFileDialog())
            {
                dialog.Filter = "CSV files (*.csv)|*.csv|All files (*.*)|*.*";
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    txtDataPath.Text = dialog.FileName;
                    LoadData(dialog.FileName);
                }
            }
        }

        // loads the file typed in the box
        private void btnLoad_Click(object sender, EventArgs e)
        {
            LoadData(txtDataPath.Text.Trim());
        }

        // reads, splits and scales the data, then clears the old results
        private void LoadData(string path)
        {
            try
            {
                data = PreparedData.Load(path, 0.70, 0.15, 42);
                net = null;
                chart.Clear();
                txtLog.Clear();
                lblPrediction.Text = "";
                lblDataInfo.Text = data.Describe();
                RebuildPredictRow();
                SetStatus("Dataset loaded. Press Train.");
            }
            catch (Exception ex)
            {
                MessageBox.Show("Could not load the dataset:" + Environment.NewLine + ex.Message,
                    "Load error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        // ---------- training ----------

        // every click makes a NEW network (new random weights) and trains it from the start
        private async void btnTrain_Click(object sender, EventArgs e)
        {
            if (data == null)
            {
                MessageBox.Show("Load a dataset first.");
                return;
            }

            // read the settings from the boxes
            int hidden = (int)numHidden.Value;
            double lrOut = (double)numLrOut.Value;
            double lrIn = (double)numLrIn.Value;
            int epochs = (int)numEpochs.Value;
            int logEvery = Math.Max(1, epochs / 100);

            // throw away the old network and the old results
            net = null;
            chart.Clear();
            txtLog.Clear();
            lblPrediction.Text = "";
            progressBar.Maximum = epochs;
            progressBar.Value = 0;
            SetBusy(true, true);
            SetStatus("Training...");
            cts = new CancellationTokenSource();
            CancellationToken token = cts.Token;

            // build the new network
            NeuralNet newNet = Trainer.BuildNetwork(data.Inputs, hidden, data.Outputs, lrOut, lrIn);
            PreparedData d = data;
            string settings = string.Format("hidden={0}, LRPOUT={1}, LRPIN={2}, epochs={3}", hidden, lrOut, lrIn, epochs);
            Log("New network: " + d.Inputs + " inputs, " + hidden + " hidden, " + d.Outputs + " outputs");
            Log("Settings: " + settings);
            Log(string.Format("{0,6} | {1,8} | {2,8} | {3,8} | {4,8}", "Epoch", "TrainMSE", "ValMSE", "TrainAcc", "ValAcc"));

            try
            {
                // train on a background thread so the window does not freeze
                List<EpochLog> logs = await Task.Run(() => Trainer.Train(
                    newNet, d.Train, d.Validation, epochs, logEvery, Environment.TickCount,
                    log => { PostToUi(() => ShowEpoch(log, epochs)); },
                    token));

                // keep the trained network and save the log file
                net = newNet;
                Trainer.SaveLog(logs, Path.Combine(ProjectPaths.FindResultsDir(), "training_log.csv"));

                // one line that sums up the training
                if (logs.Count > 0)
                {
                    EpochLog first = logs[0];
                    EpochLog last = logs[logs.Count - 1];
                    Log(Environment.NewLine + string.Format("Training MSE went from {0:F4} (epoch {1}) to {2:F4} (epoch {3}).",
                        first.TrainMse, first.Epoch, last.TrainMse, last.Epoch));
                }

                ShowEvaluation(newNet, d, settings);

                if (token.IsCancellationRequested)
                    SetStatus("Stopped early. Results are for the partly trained network.");
                else
                    SetStatus("Done. Files saved in the results folder. You can use the Predict row now.");
            }
            catch (Exception ex)
            {
                MessageBox.Show("Training failed:" + Environment.NewLine + ex.Message,
                    "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                SetStatus("Training failed.");
            }
            finally
            {
                SetBusy(false, false);
            }
        }

        // stops the training
        private void btnStop_Click(object sender, EventArgs e)
        {
            if (cts != null)
                cts.Cancel();
        }

        // adds a point to the chart and a line to the log
        private void ShowEpoch(EpochLog log, int epochs)
        {
            chart.AddPoint(log);
            progressBar.Value = Math.Min(log.Epoch, progressBar.Maximum);

            // print to the text log at epoch 1, every 100 epochs, and at the end
            if (log.Epoch == 1 || log.Epoch % 100 == 0 || log.Epoch == epochs)
                Log(string.Format("{0,6} | {1:F6} | {2:F6} | {3,8:P1} | {4,8:P1}",
                    log.Epoch, log.TrainMse, log.ValidationMse, log.TrainAccuracy, log.ValidationAccuracy));
        }

        // tests the trained network and shows and saves the results
        private void ShowEvaluation(NeuralNet trained, PreparedData d, string settings)
        {
            ClassificationReport trainReport = Evaluator.Evaluate(trained, d.Train, d.Outputs);
            ClassificationReport testReport = Evaluator.Evaluate(trained, d.Test, d.Outputs);

            // accuracy of always guessing the most common class (to compare with)
            int majorityClass;
            double baseline = Evaluator.MajorityBaselineAccuracy(d.Train, d.Test, d.Outputs, out majorityClass);

            StringWriter writer = new StringWriter();
            writer.WriteLine("Settings: " + settings);
            Evaluator.Print(trainReport, d.ClassNames, "TRAINING SET", writer);
            Evaluator.Print(testReport, d.ClassNames, "TEST SET (held out)", writer);
            writer.WriteLine();
            writer.WriteLine("Baseline (always guess '{0}'): {1:P2}", d.ClassNames[majorityClass], baseline);
            writer.WriteLine("Network test accuracy minus baseline: {0:+0.0;-0.0;0.0} percentage points",
                (testReport.Accuracy - baseline) * 100.0);

            File.WriteAllText(Path.Combine(ProjectPaths.FindResultsDir(), "evaluation_report.txt"), writer.ToString());
            Log(writer.ToString());
        }

        // ---------- hyperparameter sweep ----------

        // tries many settings, shows the best ones, and fills them into the boxes
        private async void btnSweep_Click(object sender, EventArgs e)
        {
            if (data == null)
            {
                MessageBox.Show("Load a dataset first.");
                return;
            }

            PreparedData d = data;
            string csvPath = Path.Combine(ProjectPaths.FindResultsDir(), "sweep_results.csv");
            progressBar.Maximum = 100;
            progressBar.Value = 0;
            SetBusy(true, false);
            SetStatus("Running the sweep (27 settings x 3 runs). Please wait...");
            Log("Running the hyperparameter sweep...");

            try
            {
                List<SweepResult> ranked = await Task.Run(() => HyperparameterSweep.Run(
                    d.Train, d.Validation, d.Inputs, d.Outputs, csvPath,
                    (done, total) => { PostToUi(() => ShowSweepProgress(done, total)); }));

                // show the best settings and put the winner in the boxes
                Log(HyperparameterSweep.FormatTop(ranked, 10));
                HyperParams best = ranked[0].Parameters;
                numHidden.Value = best.Hidden;
                numLrOut.Value = (decimal)best.LearningRateOutput;
                numLrIn.Value = (decimal)best.LearningRateInput;
                numEpochs.Value = best.Epochs;
                Log("The best settings were filled in above. Press Train to use them.");
                SetStatus("Sweep done. Full table saved in results/sweep_results.csv");
            }
            catch (Exception ex)
            {
                MessageBox.Show("Sweep failed:" + Environment.NewLine + ex.Message,
                    "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                SetStatus("Sweep failed.");
            }
            finally
            {
                SetBusy(false, false);
            }
        }

        // updates the progress bar during the sweep
        private void ShowSweepProgress(int done, int total)
        {
            progressBar.Maximum = total;
            progressBar.Value = Math.Min(done, total);
        }

        // ---------- predict panel ----------

        // makes the Predict row: one box for each column of the dataset, then the buttons
        private void RebuildPredictRow()
        {
            // remove the old boxes
            predictRow.Controls.Clear();
            foreach (Control c in predictDynamic)
                c.Dispose();
            predictDynamic.Clear();
            predictInputs.Clear();

            Label title = NewLabel("Predict a row:");
            predictDynamic.Add(title);
            predictRow.Controls.Add(title);

            if (data != null && data.Inputs <= MaxTypedInputs)
            {
                // one box per column, filled with the average of that column
                for (int i = 0; i < data.Inputs; i++)
                {
                    double sum = 0.0;
                    foreach (double[] row in data.Raw.Features)
                        sum += row[i];
                    double average = sum / data.Raw.Features.Count;

                    Label label = NewLabel(data.Raw.FeatureNames[i] + ":");
                    TextBox box = new TextBox();
                    box.Width = 55;
                    box.Text = average.ToString("F2", CultureInfo.InvariantCulture);

                    predictDynamic.Add(label);
                    predictDynamic.Add(box);
                    predictInputs.Add(box);
                    predictRow.Controls.Add(label);
                    predictRow.Controls.Add(box);
                }
            }
            else if (data != null)
            {
                Label note = NewLabel("(" + data.Inputs + " columns is too many to type - use the Random test row button)");
                predictDynamic.Add(note);
                predictRow.Controls.Add(note);
            }

            predictRow.Controls.Add(btnPredict);
            predictRow.Controls.Add(btnRandomRow);
            predictRow.Controls.Add(lblPrediction);
            UpdatePredictButtons();
        }

        // predicts the row typed in the boxes
        private void btnPredict_Click(object sender, EventArgs e)
        {
            if (net == null || data == null)
                return;

            // read the numbers from the boxes
            double[] raw = new double[data.Inputs];
            for (int i = 0; i < predictInputs.Count; i++)
            {
                double value;
                if (!TryParseNumber(predictInputs[i].Text, out value))
                {
                    MessageBox.Show("'" + predictInputs[i].Text + "' is not a number (" + data.Raw.FeatureNames[i] + ").");
                    return;
                }
                raw[i] = value;
            }
            ShowPrediction(raw, null);
        }

        // takes a random row from the test set, fills the boxes, and predicts it
        private void btnRandomRow_Click(object sender, EventArgs e)
        {
            if (net == null || data == null || data.TestIndices.Count == 0)
                return;

            int row = data.TestIndices[random.Next(data.TestIndices.Count)];
            double[] raw = data.Raw.Features[row];

            for (int i = 0; i < predictInputs.Count; i++)
                predictInputs[i].Text = raw[i].ToString(CultureInfo.InvariantCulture);

            ShowPrediction(raw, data.Raw.Labels[row]);
        }

        // runs one row through the trained network and shows the answer
        private void ShowPrediction(double[] raw, string actualClass)
        {
            // scale the row the same way as the training data, then run the network
            Sample sample = new Sample();
            sample.Features = data.Scaler.Transform(raw);
            sample.Target = new double[data.Outputs];
            sample.ClassIndex = 0;
            double[] outputs = Evaluator.Forward(net, sample);
            int best = Evaluator.ArgMax(outputs);

            // the score of every class
            string scores = "";
            for (int k = 0; k < outputs.Length; k++)
                scores += (k > 0 ? ", " : "") + data.ClassNames[k] + " " + outputs[k].ToString("F2");

            string text = "Predicted: " + data.ClassNames[best];
            if (actualClass != null)
                text += "   (actual: " + actualClass + (actualClass == data.ClassNames[best] ? " - correct" : " - WRONG") + ")";
            text += Environment.NewLine + "Output scores: " + scores;
            lblPrediction.Text = text;
        }

        // reads a number typed by the user (accepts 5.1 or 5,1)
        private static bool TryParseNumber(string text, out double value)
        {
            return double.TryParse(text.Trim(), NumberStyles.Float, CultureInfo.CurrentCulture, out value)
                || double.TryParse(text.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out value);
        }

        // Predict buttons only work after a network was trained
        private void UpdatePredictButtons()
        {
            bool ready = net != null && !isBusy;
            btnPredict.Enabled = ready && predictInputs.Count > 0;
            btnRandomRow.Enabled = ready;
        }

        // ---------- helpers ----------

        // adds a line to the text log
        private void Log(string text)
        {
            txtLog.AppendText(text + Environment.NewLine);
        }

        // changes the status text
        private void SetStatus(string text)
        {
            lblStatus.Text = text;
        }

        // runs a piece of code on the window thread (needed when called from the training thread)
        private void PostToUi(Action action)
        {
            if (IsHandleCreated && !IsDisposed)
                BeginInvoke(action);
        }

        // turns the buttons and boxes on or off while something is running
        private void SetBusy(bool busy, bool canStop)
        {
            isBusy = busy;
            btnTrain.Enabled = !busy;
            btnSweep.Enabled = !busy;
            btnBrowse.Enabled = !busy;
            btnLoad.Enabled = !busy;
            txtDataPath.Enabled = !busy;
            numHidden.Enabled = !busy;
            numLrOut.Enabled = !busy;
            numLrIn.Enabled = !busy;
            numEpochs.Enabled = !busy;
            btnStop.Enabled = busy && canStop;
            UpdatePredictButtons();
        }

        // stops any training when the window is closed
        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            if (cts != null)
                cts.Cancel();
            base.OnFormClosing(e);
        }
    }
}
