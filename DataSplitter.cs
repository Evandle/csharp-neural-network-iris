using System;
using System.Collections.Generic;

namespace Backprop
{
    // splits the data into train / validation / test with the same class mix in each
    public static class DataSplitter
    {
        public static void StratifiedSplit(RawDataset data, double trainFraction, double validationFraction, int seed,
            out List<int> train, out List<int> validation, out List<int> test)
        {
            var rng = new Random(seed);
            train = new List<int>();
            validation = new List<int>();
            test = new List<int>();

            foreach (string className in data.ClassNames)
            {
                // get the row numbers of this class and shuffle them
                var indices = new List<int>();
                for (int i = 0; i < data.Labels.Count; i++)
                    if (data.Labels[i] == className)
                        indices.Add(i);
                Shuffle(indices, rng);

                // how many rows go to train and validation
                int n = indices.Count;
                int nTrain = (int)Math.Round(n * trainFraction);
                int nVal = (int)Math.Round(n * validationFraction);
                if (nTrain + nVal > n)
                    nVal = n - nTrain;

                // the first rows go to train, then validation, the rest to test
                for (int i = 0; i < n; i++)
                {
                    if (i < nTrain) train.Add(indices[i]);
                    else if (i < nTrain + nVal) validation.Add(indices[i]);
                    else test.Add(indices[i]);
                }
            }
        }

        // puts a list in random order
        public static void Shuffle<T>(IList<T> list, Random rng)
        {
            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                T tmp = list[i];
                list[i] = list[j];
                list[j] = tmp;
            }
        }
    }
}
