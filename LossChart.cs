using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace Backprop
{
    // draws the loss (error) curve for the training and validation data
    public class LossChart : Control
    {
        private readonly List<EpochLog> points = new List<EpochLog>();

        public LossChart()
        {
            DoubleBuffered = true;   // stops flickering
            ResizeRedraw = true;     // redraw when the window is resized
            BackColor = Color.White;
        }

        // adds one point to the chart
        public void AddPoint(EpochLog log)
        {
            points.Add(log);
            Invalidate();
        }

        // removes all points
        public void Clear()
        {
            points.Clear();
            Invalidate();
        }

        // draws the chart
        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;

            // the area where the lines are drawn
            int left = 60, right = 15, top = 30, bottom = 35;
            Rectangle plot = new Rectangle(left, top, Width - left - right, Height - top - bottom);
            if (plot.Width < 20 || plot.Height < 20)
                return;

            using (Brush textBrush = new SolidBrush(Color.Black))
            using (Brush trainBrush = new SolidBrush(Color.RoyalBlue))
            using (Brush valBrush = new SolidBrush(Color.DarkOrange))
            using (Pen axisPen = new Pen(Color.Gray))
            using (Pen gridPen = new Pen(Color.Gainsboro))
            using (Pen trainPen = new Pen(Color.RoyalBlue, 2f))
            using (Pen valPen = new Pen(Color.DarkOrange, 2f))
            {
                // title and legend
                g.DrawString("Loss (MSE) per epoch", Font, textBrush, left, 6);
                g.FillRectangle(trainBrush, plot.Right - 190, 9, 12, 6);
                g.DrawString("Train", Font, textBrush, plot.Right - 174, 6);
                g.FillRectangle(valBrush, plot.Right - 110, 9, 12, 6);
                g.DrawString("Validation", Font, textBrush, plot.Right - 94, 6);

                g.DrawRectangle(axisPen, plot);

                // message when there is no data yet
                if (points.Count == 0)
                {
                    g.DrawString("Press Train to see the loss curve", Font, textBrush, plot.Left + 10, plot.Top + 10);
                    return;
                }

                // find the ranges of the x and y values
                double minX = points[0].Epoch;
                double maxX = points[points.Count - 1].Epoch;
                if (maxX <= minX)
                    maxX = minX + 1;
                double maxY = 0.0;
                foreach (EpochLog p in points)
                    maxY = Math.Max(maxY, Math.Max(p.TrainMse, p.ValidationMse));
                if (maxY <= 0.0)
                    maxY = 1.0;
                maxY *= 1.05;

                // turn a data value into a pixel position
                Func<double, float> toX = x => (float)(plot.Left + (x - minX) / (maxX - minX) * plot.Width);
                Func<double, float> toY = y => (float)(plot.Bottom - y / maxY * plot.Height);

                // horizontal grid lines with their values
                for (int i = 0; i <= 4; i++)
                {
                    double value = maxY * i / 4.0;
                    float py = toY(value);
                    g.DrawLine(gridPen, plot.Left, py, plot.Right, py);
                    g.DrawString(value.ToString("F3"), Font, textBrush, 4, py - 7);
                }

                // x axis labels
                string minLabel = ((int)minX).ToString();
                string maxLabel = ((int)maxX).ToString();
                SizeF maxSize = g.MeasureString(maxLabel, Font);
                g.DrawString(minLabel, Font, textBrush, plot.Left, plot.Bottom + 4);
                g.DrawString(maxLabel, Font, textBrush, plot.Right - maxSize.Width, plot.Bottom + 4);
                g.DrawString("Epoch", Font, textBrush, plot.Left + plot.Width / 2 - 15, plot.Bottom + 16);

                // the two lines
                DrawLine(g, trainPen, trainBrush, toX, toY, true);
                DrawLine(g, valPen, valBrush, toX, toY, false);
            }
        }

        // draws one line (train or validation)
        private void DrawLine(Graphics g, Pen pen, Brush brush, Func<double, float> toX, Func<double, float> toY, bool useTrain)
        {
            PointF[] pts = new PointF[points.Count];
            for (int i = 0; i < points.Count; i++)
            {
                double y = useTrain ? points[i].TrainMse : points[i].ValidationMse;
                pts[i] = new PointF(toX(points[i].Epoch), toY(y));
            }

            if (pts.Length >= 2)
                g.DrawLines(pen, pts);
            else
                g.FillEllipse(brush, pts[0].X - 3, pts[0].Y - 3, 6, 6);
        }
    }
}
