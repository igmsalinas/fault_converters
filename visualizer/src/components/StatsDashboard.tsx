import React, { useState, useEffect, useRef, useMemo } from 'react';
import Chart from 'chart.js/auto';

interface StatsDashboardProps {
  rows: Array<Record<string, string>>;
  componentsList: string[];
  metadata: any;
}

export const StatsDashboard: React.FC<StatsDashboardProps> = ({ rows, componentsList, metadata }) => {
  const [selectedComponent, setSelectedComponent] = useState(componentsList[0] || '');

  const doughnutCanvasRef = useRef<HTMLCanvasElement>(null);
  const faultsCanvasRef = useRef<HTMLCanvasElement>(null);
  const distributionCanvasRef = useRef<HTMLCanvasElement>(null);

  const doughnutChartRef = useRef<Chart | null>(null);
  const faultsChartRef = useRef<Chart | null>(null);
  const distributionChartRef = useRef<Chart | null>(null);

  // 1. Calculate General Anomaly split
  const anomalySplit = useMemo(() => {
    const normal = rows.filter(r => r.label === 'normal').length;
    const anomalous = rows.filter(r => r.label === 'anomalous').length;
    return { normal, anomalous };
  }, [rows]);

  // 2. Calculate Fault Multiplicity (0, 1, 2...)
  const faultsCounts = useMemo(() => {
    const counts: Record<number, number> = {};
    rows.forEach(r => {
      const n = parseInt(r.n_faults);
      if (!isNaN(n)) {
        counts[n] = (counts[n] || 0) + 1;
      }
    });
    const sortedKeys = Object.keys(counts).map(Number).sort((a, b) => a - b);
    return {
      labels: sortedKeys.map(k => `${k} Faults`),
      data: sortedKeys.map(k => counts[k])
    };
  }, [rows]);

  // 3. Calculate Component Multiplier Spread with custom threshold-aligned bins
  const componentDistData = useMemo(() => {
    if (!selectedComponent) return { labels: [], normalCounts: [], anomalousCounts: [], edges: [], minIdx: -1, maxIdx: -1 };

    const values = rows.map(r => parseFloat(r[selectedComponent])).filter(v => !isNaN(v));
    if (values.length === 0) return { labels: [], normalCounts: [], anomalousCounts: [], edges: [], minIdx: -1, maxIdx: -1 };

    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);

    // Fetch thresholds from metadata for selectedComponent
    const ranges = metadata?.component_ranges?.[selectedComponent];
    const T_min = ranges?.normal?.[0] ?? 0.95;
    const T_max = ranges?.normal?.[1] ?? 1.05;

    // Define customized bin edges so that T_min and T_max are exact boundaries
    const edges: number[] = [];

    // Lower region [minVal, T_min]
    if (minVal < T_min) {
      const nLower = 3;
      const step = (T_min - minVal) / nLower;
      for (let i = 0; i < nLower; i++) {
        edges.push(minVal + i * step);
      }
    }

    // Normal region [T_min, T_max]
    const nNormal = 4;
    const stepNormal = (T_max - T_min) / nNormal;
    for (let i = 0; i < nNormal; i++) {
      edges.push(T_min + i * stepNormal);
    }

    // Upper region [T_max, maxVal]
    if (maxVal > T_max) {
      const nUpper = 3;
      const stepUpper = (maxVal - T_max) / nUpper;
      for (let i = 0; i <= nUpper; i++) {
        edges.push(T_max + i * stepUpper);
      }
    } else {
      edges.push(T_max);
    }

    const numBins = edges.length - 1;
    const labels: string[] = [];
    const normalCounts = Array(numBins).fill(0);
    const anomalousCounts = Array(numBins).fill(0);

    for (let i = 0; i < numBins; i++) {
      labels.push(`${edges[i].toFixed(2)} - ${edges[i+1].toFixed(2)}`);
    }

    rows.forEach(r => {
      const val = parseFloat(r[selectedComponent]);
      if (isNaN(val)) return;

      let binIdx = -1;
      for (let i = 0; i < numBins; i++) {
        if (val >= edges[i] && val < edges[i+1]) {
          binIdx = i;
          break;
        }
      }
      
      if (val === edges[numBins]) {
        binIdx = numBins - 1;
      }

      if (binIdx === -1) {
        if (val < edges[0]) binIdx = 0;
        if (val > edges[numBins]) binIdx = numBins - 1;
      }

      if (binIdx !== -1) {
        if (r.label === 'normal') {
          normalCounts[binIdx]++;
        } else {
          anomalousCounts[binIdx]++;
        }
      }
    });

    const minIdx = edges.indexOf(T_min);
    const maxIdx = edges.indexOf(T_max);

    return { labels, normalCounts, anomalousCounts, edges, minIdx, maxIdx };
  }, [rows, selectedComponent, metadata]);

  useEffect(() => {
    // Shared chart styling
    const baseOptions: any = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: '#f3f4f6',
            font: { family: 'Plus Jakarta Sans', size: 10 }
          }
        }
      }
    };

    // Initialize Anomaly Ratio Doughnut Chart
    if (doughnutCanvasRef.current) {
      doughnutChartRef.current = new Chart(doughnutCanvasRef.current, {
        type: 'doughnut',
        data: {
          labels: ['Normal', 'Anomalous'],
          datasets: [{
            data: [anomalySplit.normal, anomalySplit.anomalous],
            backgroundColor: ['#10b981', '#ef4444'],
            borderWidth: 1,
            borderColor: 'rgba(255,255,255,0.08)'
          }]
        },
        options: {
          ...baseOptions,
          plugins: {
            ...baseOptions.plugins,
            legend: {
              ...baseOptions.plugins.legend,
              position: 'right'
            }
          }
        }
      });
    }

    // Initialize Fault Multiplicity Bar Chart
    if (faultsCanvasRef.current) {
      faultsChartRef.current = new Chart(faultsCanvasRef.current, {
        type: 'bar',
        data: {
          labels: faultsCounts.labels,
          datasets: [{
            label: 'Sample Count',
            data: faultsCounts.data,
            backgroundColor: '#6366f1',
            borderRadius: 6,
            borderWidth: 0
          }]
        },
        options: {
          ...baseOptions,
          scales: {
            x: {
              grid: { color: 'rgba(255, 255, 255, 0.04)' },
              ticks: { color: '#9ca3af', font: { family: 'Plus Jakarta Sans', size: 9 } }
            },
            y: {
              grid: { color: 'rgba(255, 255, 255, 0.04)' },
              ticks: { color: '#9ca3af', font: { family: 'Plus Jakarta Sans', size: 9 } }
            }
          },
          plugins: {
            ...baseOptions.plugins,
            legend: { display: false }
          }
        }
      });
    }

    // Custom plugin to draw vertical threshold lines exactly on the boundary edges
    const thresholdLinesPlugin = {
      id: 'thresholdLines',
      afterDraw: (chart: any) => {
        const { ctx, chartArea: { top, bottom }, scales: { x } } = chart;
        
        const drawLine = (edgeIdx: number, label: string) => {
          if (edgeIdx === -1) return;

          // The edge index corresponds to boundary (edgeIdx - 0.5) in category scales
          const tickIdx = edgeIdx - 0.5;
          
          let xPos = 0;
          if (tickIdx <= -0.5) {
            xPos = x.left;
          } else if (tickIdx >= x.ticks.length - 0.5) {
            xPos = x.right;
          } else {
            const lowerIdx = Math.floor(tickIdx);
            const upperIdx = Math.ceil(tickIdx);
            const pct = tickIdx - lowerIdx;
            
            let pLower = x.left;
            if (lowerIdx >= 0) {
              pLower = x.getPixelForValue(lowerIdx);
            } else {
              pLower = x.getPixelForValue(0) - (x.getPixelForValue(1) - x.getPixelForValue(0));
            }
            const pUpper = x.getPixelForValue(upperIdx);
            xPos = pLower + (pUpper - pLower) * pct;
          }

          // Draw vertical dashed line
          ctx.save();
          ctx.beginPath();
          ctx.setLineDash([6, 4]);
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.85)';
          ctx.lineWidth = 1.5;
          ctx.moveTo(xPos, top);
          ctx.lineTo(xPos, bottom);
          ctx.stroke();
          
          // Draw text label
          ctx.fillStyle = '#ef4444';
          ctx.font = 'bold 9px Plus Jakarta Sans';
          ctx.textAlign = 'center';
          ctx.fillText(label, xPos, top - 6);
          ctx.restore();
        };

        // Draw line at T_min and T_max index
        if (componentDistData.minIdx !== -1) {
          const T_min = componentDistData.edges[componentDistData.minIdx];
          const minVal = componentDistData.edges[0];
          const maxVal = componentDistData.edges[componentDistData.edges.length - 1];
          if (T_min > minVal && T_min < maxVal) {
            drawLine(componentDistData.minIdx, `Fault Limit: ${T_min.toFixed(2)}`);
          }
        }
        if (componentDistData.maxIdx !== -1) {
          const T_max = componentDistData.edges[componentDistData.maxIdx];
          const minVal = componentDistData.edges[0];
          const maxVal = componentDistData.edges[componentDistData.edges.length - 1];
          if (T_max > minVal && T_max < maxVal) {
            drawLine(componentDistData.maxIdx, `Fault Limit: ${T_max.toFixed(2)}`);
          }
        }
      }
    };

    // Initialize Component Distribution Clustered Bar Chart
    if (distributionCanvasRef.current) {
      distributionChartRef.current = new Chart(distributionCanvasRef.current, {
        type: 'bar',
        data: {
          labels: componentDistData.labels,
          datasets: [
            {
              label: 'Normal Class',
              data: componentDistData.normalCounts,
              backgroundColor: '#10b981',
              borderRadius: 4
            },
            {
              label: 'Anomalous Class',
              data: componentDistData.anomalousCounts,
              backgroundColor: '#ef4444',
              borderRadius: 4
            }
          ]
        },
        options: {
          ...baseOptions,
          layout: {
            padding: {
              top: 24
            }
          },
          plugins: {
            ...baseOptions.plugins,
            legend: {
              ...baseOptions.plugins.legend,
              position: 'bottom'
            }
          },
          scales: {
            x: {
              grid: { color: 'rgba(255, 255, 255, 0.04)' },
              ticks: { color: '#9ca3af', font: { family: 'Plus Jakarta Sans', size: 9 } }
            },
            y: {
              grid: { color: 'rgba(255, 255, 255, 0.04)' },
              ticks: { color: '#9ca3af', font: { family: 'Plus Jakarta Sans', size: 9 } }
            }
          }
        },
        plugins: [thresholdLinesPlugin]
      });
    }

    return () => {
      doughnutChartRef.current?.destroy();
      faultsChartRef.current?.destroy();
      distributionChartRef.current?.destroy();
    };
  }, [anomalySplit, faultsCounts, componentDistData]);

  // Update Component Distribution Chart when selected component changes
  useEffect(() => {
    if (distributionChartRef.current) {
      distributionChartRef.current.data.labels = componentDistData.labels;
      distributionChartRef.current.data.datasets[0].data = componentDistData.normalCounts;
      distributionChartRef.current.data.datasets[1].data = componentDistData.anomalousCounts;
      distributionChartRef.current.update();
    }
  }, [componentDistData]);

  return (
    <div style={styles.dashboardContainer}>
      <div style={styles.topRow}>
        <div className="chart-container glass" style={styles.cardHalf}>
          <div className="chart-header">
            <h3>Anomaly Class Balance</h3>
          </div>
          <div className="canvas-wrapper">
            <canvas ref={doughnutCanvasRef}></canvas>
          </div>
        </div>

        <div className="chart-container glass" style={styles.cardHalf}>
          <div className="chart-header">
            <h3>Fault Multiplicity Frequency</h3>
          </div>
          <div className="canvas-wrapper">
            <canvas ref={faultsCanvasRef}></canvas>
          </div>
        </div>
      </div>

      <div className="chart-container glass" style={styles.bottomCard}>
        <div className="chart-header" style={styles.bottomHeader}>
          <h3>Component Multiplier Distribution Histogram</h3>
          <div className="select-container" style={{ width: '160px' }}>
            <select
              value={selectedComponent}
              onChange={(e) => setSelectedComponent(e.target.value)}
            >
              {componentsList.map(comp => (
                <option key={comp} value={comp}>{comp}</option>
              ))}
            </select>
          </div>
        </div>
        <div className="canvas-wrapper">
          <canvas ref={distributionCanvasRef}></canvas>
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  dashboardContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
    flexGrow: 1,
    overflowY: 'auto',
    paddingBottom: '10px'
  },
  topRow: {
    display: 'flex',
    gap: '24px',
    height: '280px'
  },
  cardHalf: {
    flex: 1,
    height: '100%'
  },
  bottomCard: {
    height: '350px',
    display: 'flex',
    flexDirection: 'column'
  },
  bottomHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '10px'
  }
};
