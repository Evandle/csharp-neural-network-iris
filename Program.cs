using System;
using System.Windows.Forms;

namespace Backprop
{
    // starts the program and opens the window
    public static class Program
    {
        [STAThread]
        public static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
        }
    }
}
