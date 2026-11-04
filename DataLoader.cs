using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace Backprop
{
    // holds the data exactly as it was read from the CSV file
    public class RawDataset
    {
        public string[] FeatureNames;
        public List<double[]> Features = new List<double[]>();
        public List<string> Labels = new List<string>();
        public string[] ClassNames;      // the different class names
        public int SkippedRows;          // rows skipped because of missing values
    }

    // reads a CSV file (header row, number columns, last column = class name)
    public static class DataLoader
    {
        // reads the file and returns the data
        public static RawDataset Load(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException("Dataset not found: " + path);

            var data = new RawDataset();
            int expectedColumns = 0;

            foreach (string rawLine in File.ReadAllLines(path))
            {
                // skip empty lines
                if (string.IsNullOrWhiteSpace(rawLine))
                    continue;

                // split the line into values and remove extra spaces
                string[] parts = rawLine.Split(',');
                for (int i = 0; i < parts.Length; i++)
                    parts[i] = parts[i].Trim();

                // the first line is the header (column names)
                if (data.FeatureNames == null)
                {
                    expectedColumns = parts.Length;
                    data.FeatureNames = parts.Take(expectedColumns - 1).ToArray();
                    continue;
                }

                // skip rows with the wrong number of columns or no class name
                if (parts.Length != expectedColumns || string.IsNullOrEmpty(parts[expectedColumns - 1]))
                {
                    data.SkippedRows++;
                    continue;
                }

                // convert the feature values to numbers
                double[] features = new double[expectedColumns - 1];
                bool valid = true;
                for (int i = 0; i < features.Length; i++)
                {
                    double value;
                    // empty or non-number values count as missing
                    if (!double.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out value)
                        || double.IsNaN(value) || double.IsInfinity(value))
                    {
                        valid = false;
                        break;
                    }
                    features[i] = value;
                }

                if (!valid)
                {
                    data.SkippedRows++;
                    continue;
                }

                data.Features.Add(features);
                data.Labels.Add(parts[expectedColumns - 1]);
            }

            if (data.Features.Count == 0)
                throw new InvalidDataException("No valid rows found in " + path);

            // list of the different class names, sorted
            data.ClassNames = data.Labels.Distinct().OrderBy(x => x, StringComparer.Ordinal).ToArray();
            return data;
        }

        // turns the chosen rows into scaled samples with a one-hot answer
        public static List<Sample> ToSamples(RawDataset data, IList<int> indices, Normalizer normalizer)
        {
            var samples = new List<Sample>();
            foreach (int idx in indices)
            {
                // find the class number and set that position to 1
                int cls = Array.IndexOf(data.ClassNames, data.Labels[idx]);
                double[] target = new double[data.ClassNames.Length];
                target[cls] = 1.0;

                samples.Add(new Sample
                {
                    Features = normalizer.Transform(data.Features[idx]),
                    Target = target,
                    ClassIndex = cls
                });
            }
            return samples;
        }
    }
}
