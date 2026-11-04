using System;
using System.Collections.Generic;
using System.Linq;

namespace Backprop
{
    // the data after loading, splitting and scaling (ready for the network)
    public class PreparedData
    {
        public string SourceFile;
        public RawDataset Raw;
        public List<Sample> Train;
        public List<Sample> Validation;
        public List<Sample> Test;
        public int Inputs;          // number of input neurons
        public int Outputs;         // number of output neurons (one per class)
        public string[] ClassNames;
        public Normalizer Scaler;       // used to scale new rows the same way as the training data
        public List<int> TestIndices;   // row numbers of the test rows in Raw

        // loads the CSV, splits it, and scales it using the training rows only
        public static PreparedData Load(string csvPath, double trainFraction, double validationFraction, int seed)
        {
            var result = new PreparedData();
            result.SourceFile = csvPath;
            result.Raw = DataLoader.Load(csvPath);

            // split into train / validation / test
            List<int> trainIdx, valIdx, testIdx;
            DataSplitter.StratifiedSplit(result.Raw, trainFraction, validationFraction, seed, out trainIdx, out valIdx, out testIdx);

            // learn the min and max from the training rows only
            var normalizer = new Normalizer();
            normalizer.Fit(trainIdx.Select(i => result.Raw.Features[i]));

            result.Train = DataLoader.ToSamples(result.Raw, trainIdx, normalizer);
            result.Validation = DataLoader.ToSamples(result.Raw, valIdx, normalizer);
            result.Test = DataLoader.ToSamples(result.Raw, testIdx, normalizer);

            result.Scaler = normalizer;
            result.TestIndices = testIdx;

            result.Inputs = result.Raw.FeatureNames.Length;
            result.Outputs = result.Raw.ClassNames.Length;
            result.ClassNames = result.Raw.ClassNames;
            return result;
        }

        // short text about the data (shown in the window)
        public string Describe()
        {
            return "File: " + SourceFile + Environment.NewLine
                + "Rows: " + Raw.Features.Count + " valid (" + Raw.SkippedRows + " skipped)"
                + "   Inputs: " + Inputs
                + "   Classes (" + Outputs + "): " + string.Join(", ", ClassNames) + Environment.NewLine
                + "Split: " + Train.Count + " train / " + Validation.Count + " validation / " + Test.Count + " test";
        }
    }
}
