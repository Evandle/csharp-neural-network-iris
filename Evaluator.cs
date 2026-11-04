using System;
using System.Collections.Generic;
using System.IO;

namespace Backprop
{
    // all the results of testing the network
    public class ClassificationReport
    {
        public int[,] Confusion;     // [actual, predicted]
        public double Accuracy;
        public double[] Precision;   // per class
        public double[] Recall;      // per class
        public double[] F1;          // per class
        public double MacroPrecision;
        public double MacroRecall;
        public double MacroF1;
        public double Mse;
        public int SampleCount;
    }

    public static class Evaluator
    {
        // runs one sample through the network and returns the 3 output values
        public static double[] Forward(NeuralNet net, Sample sample)
        {
            for (int i = 0; i < sample.Features.Length; i++)
                net.setInputs(i, sample.Features[i]);

            net.run();

            double[] outputs = new double[sample.Target.Length];
            for (int k = 0; k < outputs.Length; k++)
                outputs[k] = net.getOuputData(k);
            return outputs;
        }

        // returns the position of the biggest value
        public static int ArgMax(double[] values)
        {
            int best = 0;
            for (int i = 1; i < values.Length; i++)
                if (values[i] > values[best])
                    best = i;
            return best;
        }

        // average squared error over all samples
        public static double Mse(NeuralNet net, List<Sample> samples)
        {
            double total = 0.0;
            foreach (Sample s in samples)
            {
                double[] outputs = Forward(net, s);
                double sampleLoss = 0.0;
                for (int k = 0; k < outputs.Length; k++)
                {
                    double e = s.Target[k] - outputs[k];
                    sampleLoss += e * e;
                }
                total += sampleLoss / outputs.Length;
            }
            return samples.Count == 0 ? 0.0 : total / samples.Count;
        }

        // share of samples the network gets right
        public static double Accuracy(NeuralNet net, List<Sample> samples)
        {
            if (samples.Count == 0)
                return 0.0;

            int correct = 0;
            foreach (Sample s in samples)
                if (ArgMax(Forward(net, s)) == s.ClassIndex)
                    correct++;
            return (double)correct / samples.Count;
        }

        // accuracy you would get by always guessing the most common class of the training set
        public static double MajorityBaselineAccuracy(List<Sample> train, List<Sample> test, int classCount, out int majorityClass)
        {
            // count how many training rows each class has
            int[] counts = new int[classCount];
            foreach (Sample s in train)
                counts[s.ClassIndex]++;

            majorityClass = 0;
            for (int c = 1; c < classCount; c++)
                if (counts[c] > counts[majorityClass])
                    majorityClass = c;

            // how many test rows belong to that class
            if (test.Count == 0)
                return 0.0;
            int correct = 0;
            foreach (Sample s in test)
                if (s.ClassIndex == majorityClass)
                    correct++;
            return (double)correct / test.Count;
        }

        // tests the network and calculates accuracy, precision, recall and F1
        public static ClassificationReport Evaluate(NeuralNet net, List<Sample> samples, int classCount)
        {
            var report = new ClassificationReport
            {
                Confusion = new int[classCount, classCount],
                Precision = new double[classCount],
                Recall = new double[classCount],
                F1 = new double[classCount],
                SampleCount = samples.Count,
                Mse = Evaluator.Mse(net, samples)
            };

            // count the right and wrong predictions
            int correct = 0;
            foreach (Sample s in samples)
            {
                int predicted = ArgMax(Forward(net, s));
                report.Confusion[s.ClassIndex, predicted]++;
                if (predicted == s.ClassIndex)
                    correct++;
            }
            report.Accuracy = samples.Count == 0 ? 0.0 : (double)correct / samples.Count;

            // precision, recall and F1 for each class
            for (int c = 0; c < classCount; c++)
            {
                int truePositive = report.Confusion[c, c];
                int predictedAsC = 0;   // column total
                int actuallyC = 0;      // row total
                for (int o = 0; o < classCount; o++)
                {
                    predictedAsC += report.Confusion[o, c];
                    actuallyC += report.Confusion[c, o];
                }

                report.Precision[c] = predictedAsC == 0 ? 0.0 : (double)truePositive / predictedAsC;
                report.Recall[c] = actuallyC == 0 ? 0.0 : (double)truePositive / actuallyC;
                double pr = report.Precision[c] + report.Recall[c];
                report.F1[c] = pr == 0.0 ? 0.0 : 2.0 * report.Precision[c] * report.Recall[c] / pr;

                report.MacroPrecision += report.Precision[c] / classCount;
                report.MacroRecall += report.Recall[c] / classCount;
                report.MacroF1 += report.F1[c] / classCount;
            }
            return report;
        }

        // writes the results as text (confusion matrix and the metrics table)
        public static void Print(ClassificationReport r, string[] classNames, string title, TextWriter w)
        {
            w.WriteLine();
            w.WriteLine("=== " + title + " ===");
            w.WriteLine("Samples : {0}", r.SampleCount);
            w.WriteLine("MSE     : {0:F6}", r.Mse);
            w.WriteLine("Accuracy: {0:P2}", r.Accuracy);
            w.WriteLine();

            // confusion matrix
            w.WriteLine("Confusion matrix (rows = actual, columns = predicted):");
            int width = 16;
            w.Write("".PadRight(width));
            foreach (string name in classNames)
                w.Write(Shorten(name, width - 1).PadLeft(width));
            w.WriteLine();
            for (int a = 0; a < classNames.Length; a++)
            {
                w.Write(Shorten(classNames[a], width - 1).PadRight(width));
                for (int p = 0; p < classNames.Length; p++)
                    w.Write(r.Confusion[a, p].ToString().PadLeft(width));
                w.WriteLine();
            }

            // precision, recall and F1 table
            w.WriteLine();
            w.WriteLine("{0,-18}{1,10}{2,10}{3,10}", "Class", "Precision", "Recall", "F1");
            for (int c = 0; c < classNames.Length; c++)
                w.WriteLine("{0,-18}{1,10:F3}{2,10:F3}{3,10:F3}", Shorten(classNames[c], 17), r.Precision[c], r.Recall[c], r.F1[c]);
            w.WriteLine("{0,-18}{1,10:F3}{2,10:F3}{3,10:F3}", "Macro average", r.MacroPrecision, r.MacroRecall, r.MacroF1);
        }

        // cuts a long name so the table stays neat
        private static string Shorten(string text, int maxLength)
        {
            return text.Length <= maxLength ? text : text.Substring(0, maxLength);
        }
    }
}
