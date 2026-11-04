using System;
using System.Collections.Generic;

namespace Backprop
{
    // scales every value to 0-1:  (x - min) / (max - min)
    public class Normalizer
    {
        private double[] min;
        private double[] max;

        // finds the min and max of each column (training data only)
        public void Fit(IEnumerable<double[]> rows)
        {
            min = null;
            max = null;
            foreach (double[] row in rows)
            {
                // the first row sets the starting min and max
                if (min == null)
                {
                    min = (double[])row.Clone();
                    max = (double[])row.Clone();
                    continue;
                }
                // update min and max with the other rows
                for (int i = 0; i < row.Length; i++)
                {
                    if (row[i] < min[i]) min[i] = row[i];
                    if (row[i] > max[i]) max[i] = row[i];
                }
            }
            if (min == null)
                throw new InvalidOperationException("Cannot fit the normalizer on an empty set.");
        }

        // scales one row using the saved min and max
        public double[] Transform(double[] row)
        {
            if (min == null)
                throw new InvalidOperationException("Call Fit() before Transform().");

            double[] result = new double[row.Length];
            for (int i = 0; i < row.Length; i++)
            {
                double range = max[i] - min[i];
                double scaled = range == 0.0 ? 0.0 : (row[i] - min[i]) / range;
                // keep the value between 0 and 1
                result[i] = Math.Max(0.0, Math.Min(1.0, scaled));
            }
            return result;
        }

        public double GetMin(int i) { return min[i]; }
        public double GetMax(int i) { return max[i]; }
    }
}
