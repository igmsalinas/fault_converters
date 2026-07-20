import React, { useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

interface BodeChartsProps {
  sampleData: {
    frequency: number[];
    amplitude: number[];
    phase: number[];
  } | null;
  nominalData: {
    frequency: number[];
    amplitude: number[];
    phase: number[];
  } | null;
  isAnomalous: boolean;
}

export const BodeCharts: React.FC<BodeChartsProps> = ({
  sampleData,
  nominalData,
  isAnomalous
}) => {
  const magnitudeCanvasRef = useRef<HTMLCanvasElement>(null);
  const phaseCanvasRef = useRef<HTMLCanvasElement>(null);
  
  const magnitudeChartRef = useRef<Chart | null>(null);
  const phaseChartRef = useRef<Chart | null>(null);

  useEffect(() => {
    // Initialize Charts
    const baseOptions: any = {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: 'logarithmic',
          title: {
            display: true,
            text: 'Frequency (Hz)',
            color: '#9ca3af',
            font: { family: 'Plus Jakarta Sans', size: 10 }
          },
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#9ca3af',
            font: { family: 'Plus Jakarta Sans', size: 9 },
            callback: function(value: any) {
              const dec = Math.log10(value);
              if (Math.abs(dec - Math.round(dec)) < 1e-10) {
                return value >= 1000 ? (value / 1000) + 'k' : value;
              }
              return null;
            }
          }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#9ca3af',
            font: { family: 'Plus Jakarta Sans', size: 9 }
          }
        }
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          align: 'end',
          labels: {
            color: '#f3f4f6',
            boxWidth: 15,
            font: { family: 'Plus Jakarta Sans', size: 10 }
          }
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          backgroundColor: 'rgba(10, 15, 30, 0.85)',
          titleColor: '#f3f4f6',
          bodyColor: '#9ca3af',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          bodyFont: { family: 'Plus Jakarta Sans' },
          titleFont: { family: 'Outfit', weight: 'bold' }
        }
      }
    };

    if (magnitudeCanvasRef.current) {
      magnitudeChartRef.current = new Chart(magnitudeCanvasRef.current, {
        type: 'line',
        data: {
          datasets: [
            {
              label: 'Selected Sample',
              data: [],
              borderColor: '#10b981',
              borderWidth: 2,
              pointRadius: 0,
              fill: false
            },
            {
              label: 'Nominal Reference',
              data: [],
              borderColor: 'rgba(99, 102, 241, 0.4)',
              borderWidth: 1.5,
              borderDash: [4, 4],
              pointRadius: 0,
              fill: false
            }
          ]
        },
        options: {
          ...baseOptions,
          scales: {
            ...baseOptions.scales,
            y: {
              ...baseOptions.scales.y,
              title: {
                display: true,
                text: 'Amplitude (dB)',
                color: '#9ca3af',
                font: { family: 'Plus Jakarta Sans', size: 10 }
              }
            }
          }
        }
      });
    }

    if (phaseCanvasRef.current) {
      phaseChartRef.current = new Chart(phaseCanvasRef.current, {
        type: 'line',
        data: {
          datasets: [
            {
              label: 'Selected Sample',
              data: [],
              borderColor: '#10b981',
              borderWidth: 2,
              pointRadius: 0,
              fill: false
            },
            {
              label: 'Nominal Reference',
              data: [],
              borderColor: 'rgba(99, 102, 241, 0.4)',
              borderWidth: 1.5,
              borderDash: [4, 4],
              pointRadius: 0,
              fill: false
            }
          ]
        },
        options: {
          ...baseOptions,
          scales: {
            ...baseOptions.scales,
            y: {
              ...baseOptions.scales.y,
              title: {
                display: true,
                text: 'Phase (Degrees)',
                color: '#9ca3af',
                font: { family: 'Plus Jakarta Sans', size: 10 }
              }
            }
          }
        }
      });
    }

    return () => {
      magnitudeChartRef.current?.destroy();
      phaseChartRef.current?.destroy();
    };
  }, []);

  // Update Data
  useEffect(() => {
    if (!sampleData) return;

    const themeColor = isAnomalous ? '#ef4444' : '#10b981';

    // Update Magnitude
    if (magnitudeChartRef.current) {
      const magData = sampleData.frequency.map((f, i) => ({ x: f, y: sampleData.amplitude[i] }));
      magnitudeChartRef.current.data.datasets[0].data = magData as any;
      magnitudeChartRef.current.data.datasets[0].borderColor = themeColor;
      
      if (nominalData) {
        const nominalMagData = nominalData.frequency.map((f, i) => ({ x: f, y: nominalData.amplitude[i] }));
        magnitudeChartRef.current.data.datasets[1].data = nominalMagData as any;
      } else {
        magnitudeChartRef.current.data.datasets[1].data = [];
      }
      magnitudeChartRef.current.update();
    }

    // Update Phase
    if (phaseChartRef.current) {
      const phaseData = sampleData.frequency.map((f, i) => ({ x: f, y: sampleData.phase[i] }));
      phaseChartRef.current.data.datasets[0].data = phaseData as any;
      phaseChartRef.current.data.datasets[0].borderColor = themeColor;
      
      if (nominalData) {
        const nominalPhaseData = nominalData.frequency.map((f, i) => ({ x: f, y: nominalData.phase[i] }));
        phaseChartRef.current.data.datasets[1].data = nominalPhaseData as any;
      } else {
        phaseChartRef.current.data.datasets[1].data = [];
      }
      phaseChartRef.current.update();
    }
  }, [sampleData, nominalData, isAnomalous]);

  return (
    <section className="charts-section">
      <div className="chart-container glass">
        <div className="chart-header">
          <h3>Magnitude Response</h3>
          <span className="unit-badge">dB</span>
        </div>
        <div className="canvas-wrapper">
          <canvas ref={magnitudeCanvasRef}></canvas>
        </div>
      </div>

      <div className="chart-container glass">
        <div className="chart-header">
          <h3>Phase Response</h3>
          <span className="unit-badge">Degrees</span>
        </div>
        <div className="canvas-wrapper">
          <canvas ref={phaseCanvasRef}></canvas>
        </div>
      </div>
    </section>
  );
};
