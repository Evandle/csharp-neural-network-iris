using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;

namespace Backprop
{
    // one set of settings
    public class HyperParams
    {
        public int Hidden;
        public double LearningRateOutput;
        public double LearningRateInput;
        public int Epochs;

        public override string ToString()
        {
            return string.Format("hidden={0}, LRPOUT={1}, LRPIN={2}, epochs={3}",
                Hidden, LearningRateOutput, LearningRateInput, Epochs);
        }
    }

    // the score of one set of settings
    public class SweepResult
    {
        public HyperParams Parameters;
        public double AvgValidationAccuracy;
        public double AvgValidationMse;
    }

    // tries many settings and ranks them using the validation set (never the test set)
    public static class HyperparameterSweep
    {
        private static readonly int[] HiddenSizes = { 4, 8, 16 };
        // pairs of (output learning rate, input learning rate): small, default, large
        private static readonly double[][] LearningRates =
        {
            new[] { 0.05, 0.05 },
            new[] { 0.20, 0.15 },
            new[] { 0.50, 0.50 }
        };
        private static readonly int[] EpochCounts = { 100, 300, 1000 };
        // each setting is trained 3 times because the starting weights are random
        private const int Repeats = 3;

        // runs every combination and returns the list ranked from best to worst
        public static List<SweepResult> Run(List<Sample> train, List<Sample> validation, int inputs, int outputs,
            string csvPath, Action<int, int> onProgress)
        {
            var results = new List<SweepResult>();
            int total = HiddenSizes.Length * LearningRates.Length * EpochCounts.Length;
            int done = 0;

            foreach (int hidden in HiddenSizes)
            foreach (double[] lr in LearningRates)
            foreach (int epochs in EpochCounts)
            {
                var hp = new HyperParams
                {
                    Hidden = hidden,
                    LearningRateOutput = lr[0],
                    LearningRateInput = lr[1],
                    Epochs = epochs
                };

                // train 3 times and add up the scores
                double accSum = 0.0, mseSum = 0.0;
                for (int rep = 0; rep < Repeats; rep++)
                {
                    NeuralNet net = Trainer.BuildNetwork(inputs, hp.Hidden, outputs, hp.LearningRateOutput, hp.LearningRateInput);
                    Trainer.Train(net, train, validation, hp.Epochs, hp.Epochs, 1000 + rep, null, CancellationToken.None);
                    ClassificationReport rpt = Evaluator.Evaluate(net, validation, outputs);
                    accSum += rpt.Accuracy;
                    mseSum += rpt.Mse;
                }

                // keep the average score
                results.Add(new SweepResult
                {
                    Parameters = hp,
                    AvgValidationAccuracy = accSum / Repeats,
                    AvgValidationMse = mseSum / Repeats
                });

                done++;
                if (onProgress != null)
                    onProgress(done, total);
            }

            // best = highest validation accuracy, ties go to the lower error
            List<SweepResult> ranked = results
                .OrderByDescending(r => r.AvgValidationAccuracy)
                .ThenBy(r => r.AvgValidationMse)
                .ToList();

            Save(ranked, csvPath);
            return ranked;
        }

        // makes a text table of the best settings
        public static string FormatTop(List<SweepResult> ranked, int count)
        {
            var sb = new StringBuilder();
            sb.AppendLine();
            sb.AppendLine(string.Format("Top {0} settings (average of {1} runs, scored on the validation set):", count, Repeats));
            sb.AppendLine(string.Format("{0,-4}{1,8}{2,9}{3,9}{4,8}{5,14}{6,12}", "#", "Hidden", "LRPOUT", "LRPIN", "Epochs", "Val Accuracy", "Val MSE"));
            for (int i = 0; i < Math.Min(count, ranked.Count); i++)
            {
                SweepResult r = ranked[i];
                sb.AppendLine(string.Format("{0,-4}{1,8}{2,9}{3,9}{4,8}{5,14:P2}{6,12:F5}", i + 1,
                    r.Parameters.Hidden, r.Parameters.LearningRateOutput, r.Parameters.LearningRateInput,
                    r.Parameters.Epochs, r.AvgValidationAccuracy, r.AvgValidationMse));
            }
            return sb.ToString();
        }

        // saves all the results as a CSV file
        private static void Save(List<SweepResult> ranked, string path)
        {
            using (var sw = new StreamWriter(path))
            {
                sw.WriteLine("Rank,Hidden,LRPOUT,LRPIN,Epochs,AvgValidationAccuracy,AvgValidationMSE");
                for (int i = 0; i < ranked.Count; i++)
                {
                    SweepResult r = ranked[i];
                    sw.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0},{1},{2},{3},{4},{5:F4},{6:F6}",
                        i + 1, r.Parameters.Hidden, r.Parameters.LearningRateOutput, r.Parameters.LearningRateInput,
                        r.Parameters.Epochs, r.AvgValidationAccuracy, r.AvgValidationMse));
                }
            }
        }
    }
}
