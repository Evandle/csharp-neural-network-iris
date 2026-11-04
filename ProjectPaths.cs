using System;
using System.IO;

namespace Backprop
{
    // finds the project folder, the results folder and the default dataset
    public static class ProjectPaths
    {
        // walks up from the .exe folder until it finds the .csproj file
        public static string FindProjectRoot()
        {
            DirectoryInfo dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null)
            {
                if (dir.GetFiles("*.csproj").Length > 0)
                    return dir.FullName;
                dir = dir.Parent;
            }
            return null;
        }

        // the results folder (next to the .csproj), created if it does not exist
        public static string FindResultsDir()
        {
            string root = FindProjectRoot();
            if (root == null)
                root = Directory.GetCurrentDirectory();

            string path = Path.Combine(root, "results");
            Directory.CreateDirectory(path);
            return path;
        }

        // looks for data/iris.csv in the usual places, returns null if not found
        public static string FindDefaultDataFile()
        {
            string[] candidates =
            {
                Path.Combine("data", "iris.csv"),
                Path.Combine(AppContext.BaseDirectory, "data", "iris.csv")
            };
            foreach (string c in candidates)
                if (File.Exists(c))
                    return Path.GetFullPath(c);

            string root = FindProjectRoot();
            if (root != null)
            {
                string fromRoot = Path.Combine(root, "data", "iris.csv");
                if (File.Exists(fromRoot))
                    return fromRoot;
            }
            return null;
        }
    }
}
