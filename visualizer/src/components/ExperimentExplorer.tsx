import React, { useEffect, useState, useRef, useMemo } from 'react';
import Chart from 'chart.js/auto';

interface ModelSummary {
  name: string;
  config: Record<string, any>;
  metrics: Record<string, any>;
}

interface RunSummary {
  run: string;
  models: ModelSummary[];
}

interface ExperimentExplorerProps {
  activeDatasetName?: string;
}

export const ExperimentExplorer: React.FC<ExperimentExplorerProps> = ({
  activeDatasetName = ''
}) => {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunName, setSelectedRunName] = useState<string>('');
  const [selectedModelName, setSelectedModelName] = useState<string>('');
  const [modelDetails, setModelDetails] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');
  const [rightTab, setRightTab] = useState<'history' | 'boundary' | 'faults'>('history');

  const chartCanvasRef = useRef<HTMLCanvasElement>(null);
  const chartInstanceRef = useRef<Chart | null>(null);

  // 1. Fetch runs list on mount
  useEffect(() => {
    fetch('/api/experiments')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load experiments');
        return res.json();
      })
      .then((data) => {
        setRuns(data);
        if (data.length > 0) {
          setSelectedRunName(data[0].run);
          if (data[0].models.length > 0) {
            setSelectedModelName(data[0].models[0].name);
          }
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setError('Failed to load experiments. Make sure you have trained models in the experiments/ directory.');
        setLoading(false);
      });
  }, []);

  // Filter runs based on the active dataset name
  const filteredRuns = useMemo(() => {
    if (!activeDatasetName) return runs;
    return runs.filter((r) => r.run.toLowerCase().includes(activeDatasetName.toLowerCase()));
  }, [runs, activeDatasetName]);

  // Auto-sync selected run when the filtered list changes
  useEffect(() => {
    if (filteredRuns.length > 0) {
      const isStillValid = filteredRuns.some((r) => r.run === selectedRunName);
      if (!isStillValid) {
        setSelectedRunName(filteredRuns[0].run);
        if (filteredRuns[0].models.length > 0) {
          setSelectedModelName(filteredRuns[0].models[0].name);
        }
      }
    } else {
      setSelectedRunName('');
      setSelectedModelName('');
    }
  }, [filteredRuns, selectedRunName]);

  // Get active run
  const activeRun = filteredRuns.find((r) => r.run === selectedRunName) || null;

  // 2. Fetch detailed metrics & training history when selected Run/Model changes
  useEffect(() => {
    if (!selectedRunName || !selectedModelName) return;

    fetch(`/api/experiment-details?run=${encodeURIComponent(selectedRunName)}&model=${encodeURIComponent(selectedModelName)}`)
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load experiment details');
        return res.json();
      })
      .then((data) => {
        setModelDetails(data);
      })
      .catch((err) => {
        console.error(err);
        setModelDetails(null);
      });
  }, [selectedRunName, selectedModelName]);

  // Component analysis for Explainability Tab
  const componentAnalysis = useMemo(() => {
    if (!modelDetails || !modelDetails.results) return [];

    const results = modelDetails.results;
    const stats: Record<string, { name: string; total: number; detected: number; missed: number; deviations: number[] }> = {};

    results.forEach((r: any) => {
      if (r.label === 1) { // Only analyze anomalies (faults)
        const components = r.varied_components ? r.varied_components.split(',') : [];
        components.forEach((c: string) => {
          const comp = c.trim();
          if (!comp) return;

          if (!stats[comp]) {
            stats[comp] = { name: comp, total: 0, detected: 0, missed: 0, deviations: [] };
          }
          stats[comp].total += 1;
          if (r.prediction === 1) {
            stats[comp].detected += 1;
          } else {
            stats[comp].missed += 1;
          }
          stats[comp].deviations.push(r.max_deviation);
        });
      }
    });

    return Object.values(stats).map((s) => {
      const avgDev = s.deviations.length > 0 ? (s.deviations.reduce((a, b) => a + b, 0) / s.deviations.length) * 100 : 0;
      const recall = s.total > 0 ? (s.detected / s.total) * 100 : 0;
      return {
        ...s,
        recall,
        avgDev
      };
    }).sort((a, b) => a.name.localeCompare(b.name));
  }, [modelDetails]);

  // 3. Render Chart.js when modelDetails or rightTab changes
  useEffect(() => {
    if (rightTab === 'faults' || !modelDetails || !chartCanvasRef.current) {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
      return;
    }

    const ctx = chartCanvasRef.current.getContext('2d');
    if (!ctx) return;

    // Destroy existing chart to prevent canvas reuse issue
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }

    if (rightTab === 'history') {
      const history = modelDetails.history;
      if (!history) return;
      const epochs = Object.values(history)[0] ? Array.from({ length: (Object.values(history)[0] as any).length }, (_, idx) => idx + 1) : [];

      const colors: Record<string, string> = {
        loss: '#6366f1',             // Indigo
        val_loss: '#a5b4fc',         // Light Indigo
        recon_loss: '#10b981',       // Emerald
        val_recon_loss: '#34d399',   // Light Emerald
        reconstruction_loss: '#10b981',
        val_reconstruction_loss: '#34d399',
        kl_loss: '#f59e0b',          // Amber
        val_kl_loss: '#fbbf24',      // Light Amber
        contrast_loss: '#ef4444',    // Rose
        val_contrast_loss: '#fca5a5',// Light Rose
        contrastive_loss: '#ef4444',
        val_contrastive_loss: '#fca5a5',
      };

      const datasets = Object.keys(history)
        .filter((key) => Array.isArray(history[key]) && history[key].length > 0)
        .map((key) => {
          const isVal = key.startsWith('val_');
          return {
            label: key.replace(/_/g, ' '),
            data: history[key],
            borderColor: colors[key] || (isVal ? '#9ca3af' : '#f3f4f6'),
            borderWidth: isVal ? 1.5 : 2.5,
            borderDash: isVal ? [4, 4] : [],
            pointRadius: 0,
            fill: false,
            tension: 0.1,
          };
        });

      chartInstanceRef.current = new Chart(ctx, {
        type: 'line',
        data: {
          labels: epochs,
          datasets,
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
            mode: 'index',
            intersect: false,
          },
          scales: {
            x: {
              title: {
                display: true,
                text: 'Epoch',
                color: '#9ca3af',
                font: { family: 'Plus Jakarta Sans', size: 11 },
              },
              grid: { color: 'rgba(255, 255, 255, 0.04)' },
              ticks: {
                color: '#9ca3af',
                font: { family: 'Plus Jakarta Sans', size: 10 },
              },
            },
            y: {
              title: {
                display: true,
                text: 'Loss Value',
                color: '#9ca3af',
                font: { family: 'Plus Jakarta Sans', size: 11 },
              },
              grid: { color: 'rgba(255, 255, 255, 0.04)' },
              ticks: {
                color: '#9ca3af',
                font: { family: 'Plus Jakarta Sans', size: 10 },
              },
            },
          },
          plugins: {
            legend: {
              position: 'top',
              labels: {
                color: '#f3f4f6',
                boxWidth: 12,
                font: { family: 'Plus Jakarta Sans', size: 11 },
              },
            },
            tooltip: {
              backgroundColor: 'rgba(10, 15, 30, 0.9)',
              titleColor: '#f3f4f6',
              bodyColor: '#9ca3af',
              borderColor: 'rgba(255, 255, 255, 0.1)',
              borderWidth: 1,
              bodyFont: { family: 'Plus Jakarta Sans' },
              titleFont: { family: 'Outfit', weight: 'bold' },
            },
          },
        },
      });
    } else if (rightTab === 'boundary') {
      const results = modelDetails.results || [];
      const isCarla = selectedModelName.toLowerCase().includes('carla');
      const threshold = (modelDetails.threshold && modelDetails.threshold.threshold) || 0;
      const thresholdVal = Math.max(threshold, 1e-8);

      let tnPoints = [];
      let tpPoints = [];
      let fpPoints = [];
      let fnPoints = [];
      let datasets = [];
      let xScaleConfig: any = {};
      let yScaleConfig: any = {};

      if (isCarla) {
        // Group points using PCA coordinates
        tnPoints = results.filter((r: any) => r.label === 0 && r.prediction === 0).map((r: any) => ({ x: r.pca_x ?? 0, y: r.pca_y ?? 0 }));
        tpPoints = results.filter((r: any) => r.label === 1 && r.prediction === 1).map((r: any) => ({ x: r.pca_x ?? 0, y: r.pca_y ?? 0 }));
        fpPoints = results.filter((r: any) => r.label === 0 && r.prediction === 1).map((r: any) => ({ x: r.pca_x ?? 0, y: r.pca_y ?? 0 }));
        fnPoints = results.filter((r: any) => r.label === 1 && r.prediction === 0).map((r: any) => ({ x: r.pca_x ?? 0, y: r.pca_y ?? 0 }));

        datasets = [
          {
            label: 'Normal (Detected)',
            data: tnPoints,
            backgroundColor: '#10b981', // green
            pointRadius: 6,
            pointStyle: 'circle'
          },
          {
            label: 'Fault (Detected)',
            data: tpPoints,
            backgroundColor: '#6366f1', // indigo
            pointRadius: 6,
            pointStyle: 'circle'
          },
          {
            label: 'False Alarm (FP)',
            data: fpPoints,
            backgroundColor: '#f59e0b', // amber
            pointRadius: 8,
            pointStyle: 'triangle'
          },
          {
            label: 'Missed Fault (FN)',
            data: fnPoints,
            backgroundColor: '#ef4444', // red
            pointRadius: 8,
            pointStyle: 'rectRot'
          }
        ];

        xScaleConfig = {
          title: {
            display: true,
            text: 'Latent Projection Dimension 1 (PCA)',
            color: '#9ca3af',
            font: { family: 'Plus Jakarta Sans', size: 11 },
          },
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#9ca3af',
            font: { family: 'Plus Jakarta Sans', size: 10 },
          },
        };

        yScaleConfig = {
          type: 'linear',
          title: {
            display: true,
            text: 'Latent Projection Dimension 2 (PCA)',
            color: '#9ca3af',
            font: { family: 'Plus Jakarta Sans', size: 11 },
          },
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#9ca3af',
            font: { family: 'Plus Jakarta Sans', size: 10 },
          },
        };
      } else {
        // Group points using Deviation & Score (Log Scale)
        tnPoints = results.filter((r: any) => r.label === 0 && r.prediction === 0).map((r: any) => ({ x: r.max_deviation * 100, y: Math.max(r.score, 1e-8) }));
        tpPoints = results.filter((r: any) => r.label === 1 && r.prediction === 1).map((r: any) => ({ x: r.max_deviation * 100, y: Math.max(r.score, 1e-8) }));
        fpPoints = results.filter((r: any) => r.label === 0 && r.prediction === 1).map((r: any) => ({ x: r.max_deviation * 100, y: Math.max(r.score, 1e-8) }));
        fnPoints = results.filter((r: any) => r.label === 1 && r.prediction === 0).map((r: any) => ({ x: r.max_deviation * 100, y: Math.max(r.score, 1e-8) }));

        const maxX = Math.max(...results.map((r: any) => r.max_deviation * 100), 50);

        datasets = [
          {
            label: 'Normal (Detected)',
            data: tnPoints,
            backgroundColor: '#10b981', // green
            pointRadius: 6,
            pointStyle: 'circle'
          },
          {
            label: 'Fault (Detected)',
            data: tpPoints,
            backgroundColor: '#6366f1', // indigo
            pointRadius: 6,
            pointStyle: 'circle'
          },
          {
            label: 'False Alarm (FP)',
            data: fpPoints,
            backgroundColor: '#f59e0b', // amber
            pointRadius: 8,
            pointStyle: 'triangle'
          },
          {
            label: 'Missed Fault (FN)',
            data: fnPoints,
            backgroundColor: '#ef4444', // red
            pointRadius: 8,
            pointStyle: 'rectRot'
          },
          {
            label: `Threshold (${threshold.toFixed(5)})`,
            data: [{ x: 0, y: thresholdVal }, { x: maxX, y: thresholdVal }],
            borderColor: '#ef4444',
            borderWidth: 1.5,
            borderDash: [5, 5],
            showLine: true,
            pointRadius: 0,
            fill: false,
            type: 'line' as const
          }
        ];

        xScaleConfig = {
          title: {
            display: true,
            text: 'Deviation Magnitude (%)',
            color: '#9ca3af',
            font: { family: 'Plus Jakarta Sans', size: 11 },
          },
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#9ca3af',
            font: { family: 'Plus Jakarta Sans', size: 10 },
          },
        };

        yScaleConfig = {
          type: 'logarithmic',
          title: {
            display: true,
            text: 'Anomaly Score (Log Scale)',
            color: '#9ca3af',
            font: { family: 'Plus Jakarta Sans', size: 11 },
          },
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#9ca3af',
            font: { family: 'Plus Jakarta Sans', size: 10 },
            callback: function(value: any) {
              return Number(value).toExponential(0);
            }
          },
        };
      }

      chartInstanceRef.current = new Chart(ctx, {
        type: 'scatter',
        data: {
          datasets
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: xScaleConfig,
            y: yScaleConfig
          },
          plugins: {
            legend: {
              position: 'top',
              labels: {
                color: '#f3f4f6',
                boxWidth: 10,
                font: { family: 'Plus Jakarta Sans', size: 11 },
              }
            },
            tooltip: {
              backgroundColor: 'rgba(10, 15, 30, 0.9)',
              titleColor: '#f3f4f6',
              bodyColor: '#9ca3af',
              borderColor: 'rgba(255, 255, 255, 0.1)',
              borderWidth: 1,
              callbacks: {
                label: (context) => {
                  const pt = context.raw as any;
                  return `${context.dataset.label}: Dev=${pt.x.toFixed(1)}%, Score=${pt.y.toFixed(5)}`;
                }
              }
            }
          }
        }
      });
    }

    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [modelDetails, rightTab]);

  if (loading) {
    return (
      <div className="stats-container glass" style={styles.center}>
        <div className="loading-spinner"></div>
        <p style={{ marginTop: '16px', color: 'var(--color-text-muted)' }}>Loading experiments data...</p>
      </div>
    );
  }

  if (error || runs.length === 0) {
    return (
      <div className="stats-container glass" style={styles.center}>
        <h3 style={{ color: 'var(--color-red)', marginBottom: '12px', fontFamily: 'var(--font-header)' }}>No Experiments Discovered</h3>
        <p style={{ color: 'var(--color-text-muted)', textAlign: 'center', maxWidth: '500px', lineHeight: '1.5' }}>
          {error || "We couldn't find any completed training runs in your 'experiments/' directory. Please run the training script (e.g. scripts/train_all.sh) to populate the evaluation records."}
        </p>
      </div>
    );
  }

  const activeModelDetails = modelDetails || {};
  const activeMetrics = activeModelDetails.metrics || {};
  const activeThreshold = activeModelDetails.threshold || {};

  return (
    <div style={styles.container}>
      {/* Pickers Header */}
      <div className="glass" style={styles.header}>
        <div style={styles.pickerGroup}>
          <label style={styles.label}>Experiment Run</label>
          <select
            value={selectedRunName}
            onChange={(e) => {
              setSelectedRunName(e.target.value);
              const run = filteredRuns.find((r) => r.run === e.target.value);
              if (run && run.models.length > 0) {
                setSelectedModelName(run.models[0].name);
              }
            }}
            style={styles.select}
          >
            {filteredRuns.map((r) => (
              <option key={r.run} value={r.run}>
                {r.run}
              </option>
            ))}
          </select>
        </div>

        <div style={styles.pickerGroup}>
          <label style={styles.label}>Model Architecture</label>
          <div style={styles.tabsContainer}>
            {activeRun?.models.map((m) => (
              <button
                key={m.name}
                onClick={() => setSelectedModelName(m.name)}
                style={{
                  ...styles.modelTab,
                  ...(selectedModelName === m.name ? styles.modelTabActive : {}),
                }}
              >
                {m.name}
              </button>
            ))}
          </div>
        </div>

        <div style={styles.pickerGroup}>
          <label style={styles.label}>Visual View</label>
          <div style={styles.tabsContainer}>
            <button
              onClick={() => setRightTab('history')}
              style={{
                ...styles.modelTab,
                ...(rightTab === 'history' ? styles.modelTabActive : {})
              }}
            >
              Training Progress
            </button>
            <button
              onClick={() => setRightTab('boundary')}
              style={{
                ...styles.modelTab,
                ...(rightTab === 'boundary' ? styles.modelTabActive : {})
              }}
            >
              Decision Boundary
            </button>
            <button
              onClick={() => setRightTab('faults')}
              style={{
                ...styles.modelTab,
                ...(rightTab === 'faults' ? styles.modelTabActive : {})
              }}
            >
              Fault Explainability
            </button>
          </div>
        </div>
      </div>

      {/* Main Panel Content Split */}
      <div style={styles.contentGrid}>
        
        {/* Run Overview Table & Model Metrics */}
        <div style={styles.leftCol}>
          <section className="glass" style={styles.cardSection}>
            <h3 style={styles.sectionTitle}>Run Comparison ({selectedRunName})</h3>
            <div style={styles.tableWrapper}>
              <table style={styles.table}>
                <thead>
                  <tr style={styles.tr}>
                    <th style={styles.th}>Model</th>
                    <th style={styles.th}>F1</th>
                    <th style={styles.th}>AUC-ROC</th>
                    <th style={styles.th}>Precision</th>
                    <th style={styles.th}>Recall</th>
                  </tr>
                </thead>
                <tbody>
                  {activeRun?.models.map((m) => {
                    const isSelected = m.name === selectedModelName;
                    return (
                      <tr
                        key={m.name}
                        onClick={() => setSelectedModelName(m.name)}
                        style={{
                          ...styles.rowTr,
                          ...(isSelected ? styles.rowTrActive : {}),
                        }}
                      >
                        <td style={{ ...styles.td, fontWeight: isSelected ? 700 : 500 }}>{m.name}</td>
                        <td style={styles.td}>{(m.metrics?.f1 || 0).toFixed(4)}</td>
                        <td style={styles.td}>{(m.metrics?.auc_roc || 0).toFixed(4)}</td>
                        <td style={styles.td}>{(m.metrics?.precision || 0).toFixed(4)}</td>
                        <td style={styles.td}>{(m.metrics?.recall || 0).toFixed(4)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>

          {/* Large Badge Metrics Grid */}
          <section className="glass" style={styles.cardSection}>
            <h3 style={styles.sectionTitle}>Evaluation Metrics ({selectedModelName})</h3>
            <div style={styles.metricsGrid}>
              <div className="glass-inner" style={styles.metricBadge}>
                <span style={styles.metricVal}>{(activeMetrics.f1 || 0).toFixed(4)}</span>
                <span style={styles.metricLabel}>F1-Score</span>
              </div>
              <div className="glass-inner" style={styles.metricBadge}>
                <span style={styles.metricVal}>{(activeMetrics.auc_roc || 0).toFixed(4)}</span>
                <span style={styles.metricLabel}>AUC-ROC</span>
              </div>
              <div className="glass-inner" style={styles.metricBadge}>
                <span style={styles.metricVal}>{(activeMetrics.precision || 0).toFixed(4)}</span>
                <span style={styles.metricLabel}>Precision</span>
              </div>
              <div className="glass-inner" style={styles.metricBadge}>
                <span style={styles.metricVal}>{(activeMetrics.recall || 0).toFixed(4)}</span>
                <span style={styles.metricLabel}>Recall</span>
              </div>
            </div>
            {activeMetrics.threshold_method && (
              <div style={styles.thresholdInfo}>
                <span style={{ color: 'var(--color-text-muted)' }}>Threshold Setting:</span>
                <strong style={{ color: 'var(--color-primary)', marginLeft: '8px' }}>
                  {activeMetrics.threshold_method}
                </strong>
                <span style={{ color: 'var(--color-text-muted)', marginLeft: '16px' }}>Threshold Value:</span>
                <strong style={{ color: 'var(--color-green)', marginLeft: '8px' }}>
                  {(activeThreshold.threshold || 0).toFixed(6)}
                </strong>
              </div>
            )}
          </section>
        </div>

        {/* Visualizations Panel */}
        <div style={styles.rightCol}>
          {/* Selected Tab content */}
          {rightTab === 'history' && (
            <section className="glass" style={styles.chartSectionLarge}>
              <h3 style={styles.sectionTitle}>Training Loss Progress</h3>
              <p style={styles.tabDescription}>
                Visualizes the training and validation loss values across training epochs. Use this to monitor convergence and detect overfitting.
              </p>
              <div style={styles.chartContainerLarge}>
                {modelDetails && modelDetails.history ? (
                  <canvas ref={chartCanvasRef} />
                ) : (
                  <div style={styles.noHistory}>
                    <span style={{ color: 'var(--color-text-muted)' }}>No training history records available for this model.</span>
                  </div>
                )}
              </div>
            </section>
          )}

          {rightTab === 'boundary' && (
            <section className="glass" style={styles.chartSectionLarge}>
              <h3 style={styles.sectionTitle}>
                {selectedModelName.toLowerCase().includes('carla') ? 'Contrastive Latent Space Projection (PCA)' : 'Decision Boundary & Classification Spread'}
              </h3>
              <p style={styles.tabDescription}>
                {selectedModelName.toLowerCase().includes('carla')
                  ? 'Visualizes individual test samples projected into the model\'s 2D contrastive latent space using PCA. Normal converter sweeps cluster tightly together, while physical converter anomalies are projected outward.'
                  : 'Visualizes individual test samples projected by their anomaly score vs component deviation magnitude. High-scoring samples exceeding the threshold (dashed red line) are predicted as anomalies.'
                }
              </p>
              <div style={styles.chartContainerLarge}>
                {modelDetails && modelDetails.results ? (
                  <canvas ref={chartCanvasRef} />
                ) : (
                  <div style={styles.noHistory}>
                    <span style={{ color: 'var(--color-text-muted)' }}>No test results CSV data available. Make sure this model is evaluated.</span>
                  </div>
                )}
              </div>
            </section>
          )}

          {rightTab === 'faults' && (
            <section className="glass" style={styles.cardSection}>
              <h3 style={styles.sectionTitle}>Physical Component Fault Analysis</h3>
              <p style={styles.tabDescription}>
                Breakdown of anomaly detection performance (Recall) across specific power electronic component failures. This shows which parameter variations are hardest for the model to isolate.
              </p>
              <div style={styles.faultsList}>
                {componentAnalysis.length > 0 ? (
                  componentAnalysis.map((comp) => {
                    const isLowRecall = comp.recall < 85;
                    const progressColor = comp.recall > 95 ? 'var(--color-green)' : comp.recall > 85 ? 'var(--color-primary)' : 'var(--color-red)';
                    return (
                      <div key={comp.name} className="glass-inner" style={styles.faultCard}>
                        <div style={styles.faultHeader}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ fontSize: '18px' }}>⚡</span>
                            <span style={styles.faultName}>Component parameter: <strong>{comp.name}</strong></span>
                          </div>
                          <span style={{ ...styles.faultRecallBadge, backgroundColor: progressColor + '20', color: progressColor }}>
                            Recall: {comp.recall.toFixed(1)}%
                          </span>
                        </div>

                        <div style={styles.progressBarBg}>
                          <div style={{ ...styles.progressBarFill, width: `${comp.recall}%`, backgroundColor: progressColor }} />
                        </div>

                        <div style={styles.faultDetailsRow}>
                          <span>Detected: <strong>{comp.detected}</strong> / {comp.total} samples</span>
                          <span>Avg variation: <strong>{comp.avgDev.toFixed(1)}%</strong></span>
                          <span>Missed: <strong style={{ color: comp.missed > 0 ? 'var(--color-red)' : 'var(--color-text-main)' }}>{comp.missed}</strong></span>
                        </div>

                        {isLowRecall && (
                          <div style={styles.faultAlert} className="glass-inner">
                            ⚠️ <strong>Critical Gap:</strong> Missed {comp.missed} anomalies in {comp.name} parameter sweeps. This model struggles with smaller deviations on this specific component.
                          </div>
                        )}
                      </div>
                    );
                  })
                ) : (
                  <div style={styles.noHistory}>
                    <span style={{ color: 'var(--color-text-muted)' }}>No components fault data available. Ensure test_results.csv is generated during evaluation.</span>
                  </div>
                )}
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    height: '100%',
    overflowY: 'auto',
    paddingRight: '4px',
  },
  center: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '400px',
  },
  header: {
    display: 'flex',
    flexWrap: 'wrap',
    padding: '16px 20px',
    gap: '24px',
    alignItems: 'center',
    borderRadius: 'var(--border-radius-lg)',
  },
  pickerGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  label: {
    fontFamily: 'var(--font-header)',
    fontSize: '11px',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '1px',
    color: 'var(--color-text-muted)',
  },
  select: {
    background: '#111625',
    border: '1px solid var(--border-glass)',
    color: 'var(--color-text-main)',
    padding: '8px 16px',
    borderRadius: 'var(--border-radius-md)',
    fontFamily: 'var(--font-body)',
    fontSize: '13px',
    cursor: 'pointer',
    outline: 'none',
    width: '220px',
    appearance: 'auto',
  },
  tabsContainer: {
    display: 'flex',
    gap: '6px',
    flexWrap: 'wrap',
  },
  modelTab: {
    background: 'rgba(255, 255, 255, 0.03)',
    border: '1px solid var(--border-glass)',
    color: 'var(--color-text-muted)',
    padding: '6px 12px',
    borderRadius: 'var(--border-radius-sm)',
    fontSize: '12px',
    fontWeight: 600,
    cursor: 'pointer',
    fontFamily: 'var(--font-body)',
    transition: 'var(--transition-smooth)',
  },
  modelTabActive: {
    background: 'var(--color-primary-glow)',
    borderColor: 'var(--color-primary)',
    color: 'var(--color-text-main)',
  },
  contentGrid: {
    display: 'grid',
    gridTemplateColumns: 'minmax(400px, 1fr) minmax(500px, 1.25fr)',
    gap: '16px',
    alignItems: 'stretch',
  },
  leftCol: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  rightCol: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    height: '100%',
  },
  tabDescription: {
    fontSize: '13px',
    color: 'var(--color-text-muted)',
    lineHeight: '1.5',
    marginBottom: '20px',
  },
  cardSection: {
    padding: '20px',
    borderRadius: 'var(--border-radius-lg)',
  },
  chartSection: {
    padding: '20px',
    borderRadius: 'var(--border-radius-lg)',
    height: '350px',
    display: 'flex',
    flexDirection: 'column',
  },
  chartSectionLarge: {
    padding: '20px',
    borderRadius: 'var(--border-radius-lg)',
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    minHeight: '520px',
  },
  sectionTitle: {
    fontFamily: 'var(--font-header)',
    fontSize: '16px',
    fontWeight: 700,
    marginBottom: '16px',
    letterSpacing: '-0.2px',
  },
  tableWrapper: {
    overflowX: 'auto',
    borderRadius: 'var(--border-radius-md)',
    border: '1px solid var(--border-glass)',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '13px',
    textAlign: 'left',
  },
  tr: {
    background: 'rgba(255, 255, 255, 0.02)',
    borderBottom: '1px solid var(--border-glass)',
  },
  th: {
    padding: '10px 14px',
    fontFamily: 'var(--font-header)',
    fontWeight: 600,
    color: 'var(--color-text-muted)',
    fontSize: '11px',
    textTransform: 'uppercase',
  },
  rowTr: {
    borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
    cursor: 'pointer',
    transition: 'var(--transition-smooth)',
  },
  rowTrActive: {
    background: 'rgba(99, 102, 241, 0.06)',
  },
  td: {
    padding: '10px 14px',
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '12px',
  },
  metricBadge: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '16px',
    borderRadius: 'var(--border-radius-md)',
  },
  metricVal: {
    fontFamily: 'var(--font-header)',
    fontSize: '26px',
    fontWeight: 800,
    color: 'var(--color-text-main)',
    letterSpacing: '-0.5px',
  },
  metricLabel: {
    fontSize: '10px',
    textTransform: 'uppercase',
    color: 'var(--color-text-muted)',
    fontWeight: 600,
    marginTop: '4px',
    letterSpacing: '0.5px',
  },
  thresholdInfo: {
    marginTop: '16px',
    paddingTop: '16px',
    borderTop: '1px solid var(--border-glass)',
    fontSize: '12px',
    display: 'flex',
    alignItems: 'center',
  },
  chartContainer: {
    flex: 1,
    position: 'relative',
    height: '100%',
    width: '100%',
  },
  chartContainerLarge: {
    flex: 1,
    position: 'relative',
    height: '100%',
    width: '100%',
  },
  noHistory: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    fontSize: '13px',
  },
  hyperparamsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
    gap: '10px',
  },
  hCard: {
    display: 'flex',
    flexDirection: 'column',
    padding: '10px 14px',
    borderRadius: 'var(--border-radius-md)',
  },
  hKey: {
    fontSize: '9px',
    textTransform: 'uppercase',
    color: 'var(--color-text-muted)',
    fontWeight: 600,
    letterSpacing: '0.5px',
  },
  hVal: {
    fontSize: '12px',
    fontWeight: 600,
    marginTop: '4px',
    wordBreak: 'break-all',
  },
  faultsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  faultCard: {
    padding: '16px',
    borderRadius: 'var(--border-radius-md)',
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  },
  faultHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  faultName: {
    fontSize: '14px',
    color: 'var(--color-text-main)',
  },
  faultRecallBadge: {
    fontSize: '12px',
    fontWeight: 700,
    padding: '4px 10px',
    borderRadius: '12px',
  },
  progressBarBg: {
    width: '100%',
    height: '8px',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '4px',
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    borderRadius: '4px',
    transition: 'width 0.6s ease',
  },
  faultDetailsRow: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '12px',
    color: 'var(--color-text-muted)',
  },
  faultAlert: {
    padding: '10px 12px',
    borderRadius: 'var(--border-radius-sm)',
    fontSize: '12px',
    color: '#ef4444',
    border: '1px solid rgba(239, 68, 68, 0.1)',
    backgroundColor: 'rgba(239, 68, 68, 0.02)',
    lineHeight: '1.4',
  },
};
