using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading;

namespace Backprop
{
    // one line of the training log
    public class EpochLog
    {
        public int Epoch;
        public double TrainMse;
        public double ValidationMse;
        public double TrainAccuracy;
        public double ValidationAccuracy;
    }

    // creates the network and runs the training loop
    public static class Trainer
    {
        // makes a NEW network with new random weights
        public static NeuralNet BuildNetwork(int inputs, int hidden, int outputs, double learningRateOutput, double learningRateInput)
        {
            var net = new NeuralNet(inputs, hidden, outputs);
            net.LRPOUT = learningRateOutput;   // hidden -> output learning rate
            net.LRPIN = learningRateInput;     // input -> hidden learning rate
            return net;
        }

        // trains the network one sample at a time (backpropagation)
        // onLog is called each time a log line is made (can be null), cancelToken stops the training early
        public static List<EpochLog> Train(NeuralNet net, List<Sample> train, List<Sample> validation,
            int epochs, int logEvery, int shuffleSeed, Action<EpochLog> onLog, CancellationToken cancelToken)
        {
            var logs = new List<EpochLog>();
            var rng = new Random(shuffleSeed);
            var order = new List<Sample>(train);

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                // stop if the Stop button was pressed
                if (cancelToken.IsCancellationRequested)
                    break;

                // new random order every epoch
                DataSplitter.Shuffle(order, rng);
                double totalLoss = 0.0;
                int correctCount = 0;

                foreach (Sample s in order)
                {
                    // give the network the inputs and the expected outputs
                    for (int i = 0; i < s.Features.Length; i++)
                        net.setInputs(i, s.Features[i]);
                    for (int k = 0; k < s.Target.Length; k++)
                        net.setDesiredOutput(k, s.Target[k]);

                    // forward pass, then update the weights (backpropagation)
                    net.learn();

                    // the outputs are still from the forward pass, so use them for the loss and the accuracy
                    double sampleLoss = 0.0;
                    int predicted = 0;
                    for (int k = 0; k < s.Target.Length; k++)
                    {
                        double e = s.Target[k] - net.getOuputData(k);
                        sampleLoss += e * e;
                        if (net.getOuputData(k) > net.getOuputData(predicted))
                            predicted = k;
                    }
                    totalLoss += sampleLoss / s.Target.Length;
                    if (predicted == s.ClassIndex)
                        correctCount++;
                }

                // save a log line at epoch 1, every logEvery epochs, and at the last epoch
                if (epoch == 1 || epoch % logEvery == 0 || epoch == epochs)
                {
                    var log = new EpochLog
                    {
                        Epoch = epoch,
                        TrainMse = totalLoss / order.Count,
                        ValidationMse = Evaluator.Mse(net, validation),
                        TrainAccuracy = (double)correctCount / order.Count,
                        ValidationAccuracy = Evaluator.Accuracy(net, validation)
                    };
                    logs.Add(log);

                    if (onLog != null)
                        onLog(log);
                }
            }
            return logs;
        }

        // saves the log as a CSV file
        public static void SaveLog(List<EpochLog> logs, string path)
        {
            using (var sw = new StreamWriter(path))
            {
                sw.WriteLine("Epoch,TrainMSE,ValidationMSE,TrainAccuracy,ValidationAccuracy");
                foreach (EpochLog l in logs)
                    sw.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0},{1:F6},{2:F6},{3:F4},{4:F4}",
                        l.Epoch, l.TrainMse, l.ValidationMse, l.TrainAccuracy, l.ValidationAccuracy));
            }
        }
    }
}
