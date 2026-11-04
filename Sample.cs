namespace Backprop
{
    // one row of data (the inputs and the expected answer)
    public class Sample
    {
        public double[] Features;   // input values scaled to 0-1
        public double[] Target;     // expected output, e.g. {0,1,0}
        public int ClassIndex;      // number of the correct class
    }
}
