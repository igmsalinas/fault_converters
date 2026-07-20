import { useState, useEffect, useMemo } from 'react';
import { FolderPicker } from './components/FolderPicker';
import { Sidebar } from './components/Sidebar';
import { BodeCharts } from './components/BodeCharts';
import { ParameterGrid } from './components/ParameterGrid';
import { StatsDashboard } from './components/StatsDashboard';
import { ExperimentExplorer } from './components/ExperimentExplorer';

export default function App() {
  const [folderName, setFolderName] = useState('');
  const [fileMap, setFileMap] = useState<Map<string, File>>(new Map());
  const [manifestHeaders, setManifestHeaders] = useState<string[]>([]);
  const [manifestRows, setManifestRows] = useState<Array<Record<string, string>>>([]);
  
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  
  const [selectedFilename, setSelectedFilename] = useState('');
  const [selectedSampleData, setSelectedSampleData] = useState<any>(null);
  const [nominalData, setNominalData] = useState<any>(null);

  // API Mode States
  const [apiDatasets, setApiDatasets] = useState<Array<{ name: string; converter: string; path: string }>>([]);
  const [selectedDatasetPath, setSelectedDatasetPath] = useState('');
  const [isApiMode, setIsApiMode] = useState(false);

  // dataset.json metadata states
  const [datasetMetadata, setDatasetMetadata] = useState<any>(null);
  const [selectedComponentFilter, setSelectedComponentFilter] = useState('');

  // SPA active tab selection
  const [activeTab, setActiveTab] = useState<'viewer' | 'stats' | 'experiments'>('viewer');

  const activeDatasetName = useMemo(() => {
    if (!isApiMode || !selectedDatasetPath) return '';
    const ds = apiDatasets.find((d) => d.path === selectedDatasetPath);
    return ds ? ds.name : '';
  }, [isApiMode, selectedDatasetPath, apiDatasets]);

  // 1. Autoexplore datasets via Vite dev server middleware on mount
  useEffect(() => {
    fetch('/api/datasets')
      .then((res) => {
        if (!res.ok) throw new Error('API not available');
        return res.json();
      })
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setApiDatasets(data);
          setIsApiMode(true);
          setSelectedDatasetPath(data[0].path);
        }
      })
      .catch((err) => {
        console.log('Vite API not running or returned error. Falling back to local FolderPicker.', err);
      });
  }, []);

  // 2. Fetch manifest & pre-load nominal reference file when selectedDatasetPath changes (API Mode)
  useEffect(() => {
    if (!isApiMode || !selectedDatasetPath) return;

    fetch(`/api/manifest?path=${encodeURIComponent(selectedDatasetPath)}`)
      .then((res) => res.json())
      .then(async (manifest) => {
        setManifestHeaders(manifest.headers);
        setManifestRows(manifest.rows);
        setDatasetMetadata(manifest.metadata || null);
        setSelectedComponentFilter(''); // Reset component filter
        
        // Find folder details
        const dsInfo = apiDatasets.find((d) => d.path === selectedDatasetPath);
        setFolderName(dsInfo ? dsInfo.name : 'Unknown');

        // Find nominal reference file
        const nominalFile = findNominalFilename(manifest.rows, manifest.headers);
        if (nominalFile) {
          try {
            const resNom = await fetch(`/api/sample?path=${encodeURIComponent(selectedDatasetPath)}&filename=${nominalFile}`);
            setNominalData(await resNom.json());
          } catch (e) {
            console.error('Failed to fetch nominal sample:', e);
            setNominalData(null);
          }
        } else {
          setNominalData(null);
        }

        // Auto-select first sample
        if (manifest.rows.length > 0) {
          setSelectedFilename(manifest.rows[0].filename);
        } else {
          setSelectedFilename('');
        }
      })
      .catch((err) => {
        console.error('Error fetching manifest:', err);
      });
  }, [selectedDatasetPath, isApiMode, apiDatasets]);

  // Parse CSV Helper
  const parseCSV = (text: string) => {
    const lines = text.split(/\r?\n/);
    if (lines.length === 0) return { headers: [], rows: [] };
    
    const parseLine = (line: string) => {
      const result: string[] = [];
      let cur = '';
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          result.push(cur.trim());
          cur = '';
        } else {
          cur += char;
        }
      }
      result.push(cur.trim());
      return result;
    };
    
    const headers = parseLine(lines[0]);
    const rows: Array<Record<string, string>> = [];
    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      const values = parseLine(lines[i]);
      if (values.length === headers.length) {
        const row: Record<string, string> = {};
        for (let j = 0; j < headers.length; j++) {
          row[headers[j]] = values[j];
        }
        rows.push(row);
      }
    }
    return { headers, rows };
  };

  // Parse TXT Helper
  const parseSampleTXT = (text: string) => {
    const frequencies: number[] = [];
    const amplitudes: number[] = [];
    const phases: number[] = [];
    
    const lines = text.split(/\r?\n/);
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      const parts = line.split(/\s+/);
      if (parts.length >= 3) {
        frequencies.push(parseFloat(parts[0]));
        amplitudes.push(parseFloat(parts[1]));
        phases.push(parseFloat(parts[2]));
      }
    }
    
    return {
      frequency: frequencies,
      amplitude: amplitudes,
      phase: phases
    };
  };

  // File Reader Promise
  const readFileAsText = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.onerror = (err) => reject(err);
      reader.readAsText(file);
    });
  };

  // Handle Directory Selection (Local mode)
  const handleDirectorySelected = async (files: FileList) => {
    const map = new Map<string, File>();
    let manifestFile: File | null = null;
    let metadataFile: File | null = null;
    let name = '';

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const parts = file.webkitRelativePath.split('/');
      if (!name && parts.length > 0) {
        name = parts[0];
      }
      const filename = parts[parts.length - 1];
      map.set(filename, file);
      
      if (filename.toLowerCase().startsWith('manifest') && filename.toLowerCase().endsWith('.csv')) {
        if (!manifestFile || filename === 'manifest.csv') {
          manifestFile = file;
        }
      }
      if (filename === 'dataset.json') {
        metadataFile = file;
      }
    }

    if (!manifestFile) {
      alert("Error: manifest.csv (or manifest_*.csv) not found in the selected folder. Please select a valid dataset folder.");
      return;
    }

    try {
      const csvText = await readFileAsText(manifestFile);
      const manifest = parseCSV(csvText);
      
      let metadata = null;
      if (metadataFile) {
        try {
          const jsonText = await readFileAsText(metadataFile);
          metadata = JSON.parse(jsonText);
        } catch (e) {
          console.error('Error parsing dataset.json:', e);
        }
      }
      
      setFileMap(map);
      setFolderName(name);
      setManifestHeaders(manifest.headers);
      setManifestRows(manifest.rows);
      setDatasetMetadata(metadata);
      setSelectedComponentFilter(''); // Reset component filter
      
      // Find nominal reference file
      const nominalFile = findNominalFilename(manifest.rows, manifest.headers);
      
      if (nominalFile) {
        const fileObj = map.get(nominalFile);
        if (fileObj) {
          const txt = await readFileAsText(fileObj);
          setNominalData(parseSampleTXT(txt));
        }
      }
      
      // Auto-select first sample
      if (manifest.rows.length > 0) {
        setSelectedFilename(manifest.rows[0].filename);
      }
    } catch (err) {
      console.error(err);
      alert('Error parsing manifest CSV.');
    }
  };

  // Find nominal filename
  const findNominalFilename = (rows: Array<Record<string, string>>, headers: string[]) => {
    const compCols = headers.filter((h) => !['filename', 'set', 'label', 'n_faults', 'mode', 'key'].includes(h));
    let bestFilename = '';
    let minDev = Infinity;
    
    for (const row of rows) {
      if (row.set === 'healthy' || row.label === 'normal') {
        let devSum = 0;
        for (const col of compCols) {
          const val = parseFloat(row[col]) || 1.0;
          devSum += Math.abs(val - 1.0);
        }
        if (devSum < minDev) {
          minDev = devSum;
          bestFilename = row.filename;
        }
        if (devSum === 0) break;
      }
    }
    return bestFilename || (rows.length > 0 ? rows[0].filename : '');
  };

  // Helper to determine if a specific component's value is anomalous
  const isComponentAnomalous = (comp: string, row: Record<string, string>, metadata: any) => {
    const val = parseFloat(row[comp]);
    if (isNaN(val)) return false;

    if (!metadata || !metadata.component_ranges || !metadata.component_ranges[comp]) {
      // Fallback: 5% deviation threshold
      return Math.abs(val - 1.0) > 0.05;
    }

    const ranges = metadata.component_ranges[comp];
    
    // Check if outside normal range
    if (ranges.normal && Array.isArray(ranges.normal)) {
      const [minNormal, maxNormal] = ranges.normal;
      if (val < minNormal || val > maxNormal) {
        return true;
      }
    }

    // Check if inside anomalous range
    if (ranges.anomalous && Array.isArray(ranges.anomalous)) {
      const [minAnom, maxAnom] = ranges.anomalous;
      if (val >= minAnom && val <= maxAnom) {
        return true;
      }
    }

    return false;
  };

  // Components list memo
  const componentsList = useMemo(() => {
    if (datasetMetadata && Array.isArray(datasetMetadata.components)) {
      return datasetMetadata.components;
    }
    return manifestHeaders.filter((h) => !['filename', 'set', 'label', 'n_faults', 'mode', 'key'].includes(h));
  }, [datasetMetadata, manifestHeaders]);

  // Filter and Search rows memo
  const filteredRows = useMemo(() => {
    return manifestRows.filter((row) => {
      // Label filter
      if (filter === 'normal' && row.label !== 'normal') return false;
      if (filter === 'anomalous' && row.label !== 'anomalous') return false;
      
      // Component-specific anomaly filter
      if (filter === 'anomalous' && selectedComponentFilter) {
        if (!isComponentAnomalous(selectedComponentFilter, row, datasetMetadata)) {
          return false;
        }
      }

      // Search query filter
      if (searchQuery) {
        const fn = row.filename.toLowerCase();
        if (!fn.includes(searchQuery.toLowerCase())) return false;
      }
      return true;
    });
  }, [manifestRows, filter, selectedComponentFilter, searchQuery, datasetMetadata]);

  // Selected row memo
  const selectedRow = useMemo(() => {
    return manifestRows.find((r) => r.filename === selectedFilename) || null;
  }, [manifestRows, selectedFilename]);

  // 3. Load selected sample data when selectedFilename or mode changes
  useEffect(() => {
    if (!selectedFilename) return;

    if (isApiMode) {
      fetch(`/api/sample?path=${encodeURIComponent(selectedDatasetPath)}&filename=${selectedFilename}`)
        .then((res) => res.json())
        .then((data) => {
          setSelectedSampleData(data);
        })
        .catch((err) => {
          console.error('Error loading sample data via API:', err);
        });
    } else {
      if (fileMap.size === 0) return;
      const fileObj = fileMap.get(selectedFilename);
      if (!fileObj) return;

      readFileAsText(fileObj)
        .then((txt) => {
          setSelectedSampleData(parseSampleTXT(txt));
        })
        .catch((err) => {
          console.error('Error loading sample data locally:', err);
        });
    }
  }, [selectedFilename, fileMap, isApiMode, selectedDatasetPath]);

  // Sidebar stats memo
  const stats = useMemo(() => {
    const total = manifestRows.length;
    const normal = manifestRows.filter((r) => r.label === 'normal').length;
    const anomalous = manifestRows.filter((r) => r.label === 'anomalous').length;
    return { total, normal, anomalous };
  }, [manifestRows]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (document.activeElement?.tagName === 'INPUT') return;
      if (filteredRows.length === 0) return;

      const idx = filteredRows.findIndex((r) => r.filename === selectedFilename);
      if (e.key === 'ArrowLeft' && idx > 0) {
        setSelectedFilename(filteredRows[idx - 1].filename);
      } else if (e.key === 'ArrowRight' && idx !== -1 && idx < filteredRows.length - 1) {
        setSelectedFilename(filteredRows[idx + 1].filename);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [filteredRows, selectedFilename]);

  // Next / Prev button click handlers
  const handlePrev = () => {
    const idx = filteredRows.findIndex((r) => r.filename === selectedFilename);
    if (idx > 0) {
      setSelectedFilename(filteredRows[idx - 1].filename);
    }
  };

  const handleNext = () => {
    const idx = filteredRows.findIndex((r) => r.filename === selectedFilename);
    if (idx !== -1 && idx < filteredRows.length - 1) {
      setSelectedFilename(filteredRows[idx + 1].filename);
    }
  };

  // Change directory handler
  const handleChangeDirectory = () => {
    setIsApiMode(false);
    setFolderName('');
    setFileMap(new Map());
    setManifestHeaders([]);
    setManifestRows([]);
    setFilter('all');
    setSearchQuery('');
    setSelectedFilename('');
    setSelectedSampleData(null);
    setNominalData(null);
    setDatasetMetadata(null);
    setSelectedComponentFilter('');
    setActiveTab('viewer');
  };

  // Reset component filter when label filter changes to non-anomalous
  useEffect(() => {
    if (filter !== 'anomalous') {
      setSelectedComponentFilter('');
    }
  }, [filter]);

  const isLoaded = isApiMode || fileMap.size > 0;

  return (
    <>
      {/* Background Gradients */}
      <div className="bg-gradient bg-violet"></div>
      <div className="bg-gradient bg-indigo"></div>

      {!isLoaded ? (
        <FolderPicker onDirectorySelected={handleDirectorySelected} />
      ) : (
        <div style={styles.appLayout}>
          {/* Top Navbar */}
          <nav style={styles.topNavbar} className="glass">
            {/* Logo */}
            <div style={styles.navLogo}>
              <span style={styles.navLogoIcon}>⚡</span>
              <div style={styles.navLogoText}>
                <h1 style={styles.navLogoTitle}>FaultConverter</h1>
                <p style={styles.navLogoSubtitle}>Dataset Visualization Suite</p>
              </div>
            </div>

            {/* SPA Tab Bar Switcher */}
            <div style={styles.tabBar} className="glass-inner">
              <button
                style={{
                  ...styles.tabBtn,
                  ...(activeTab === 'viewer' ? styles.tabBtnActive : {})
                }}
                onClick={() => setActiveTab('viewer')}
              >
                Bode Viewer
              </button>
              <button
                style={{
                  ...styles.tabBtn,
                  ...(activeTab === 'stats' ? styles.tabBtnActive : {})
                }}
                onClick={() => setActiveTab('stats')}
              >
                Dataset Dashboard
              </button>
              <button
                style={{
                  ...styles.tabBtn,
                  ...(activeTab === 'experiments' ? styles.tabBtnActive : {})
                }}
                onClick={() => setActiveTab('experiments')}
              >
                Experiment Explorer
              </button>
            </div>

            {/* Global Dataset Selector (Top Right) */}
            {isApiMode && apiDatasets.length > 0 ? (
              <div style={styles.globalDatasetContainer} className="glass-inner">
                <span style={styles.globalDatasetLabel}>Select Dataset</span>
                <select
                  value={selectedDatasetPath}
                  onChange={(e) => setSelectedDatasetPath(e.target.value)}
                  style={styles.globalDatasetSelect}
                >
                  {apiDatasets.map((ds) => (
                    <option key={ds.path} value={ds.path}>
                      [{ds.converter.toUpperCase()}] {ds.name}
                    </option>
                  ))}
                </select>
              </div>
            ) : (
              <div style={{ width: '150px' }} />
            )}
          </nav>

          {/* Bottom Layout Split (Sidebar + Content or Full Content) */}
          <div style={styles.mainContainer}>
            {activeTab === 'viewer' ? (
              <>
                <Sidebar
                  stats={stats}
                  filter={filter}
                  setFilter={setFilter}
                  searchQuery={searchQuery}
                  setSearchQuery={setSearchQuery}
                  samples={filteredRows}
                  selectedFilename={selectedFilename}
                  onSelectSample={setSelectedFilename}
                  onChangeDirectory={handleChangeDirectory}
                  componentsList={componentsList}
                  selectedComponentFilter={selectedComponentFilter}
                  onSelectComponentFilter={setSelectedComponentFilter}
                />
                
                <main className="main-content">
                  <header className="viewer-header glass">
                    <div className="header-info">
                      <div className="title-row">
                        <h2 id="current-sample-id">{selectedFilename.replace('.txt', '')}</h2>
                        {selectedRow && (
                          <span className={`badge ${selectedRow.label}`}>
                            {selectedRow.label}
                          </span>
                        )}
                      </div>
                      <p className="dataset-info">
                        Dataset: <span id="current-dataset-name">{folderName}</span> | Mode: <span style={{ textTransform: 'uppercase', fontWeight: 600 }}>{isApiMode ? 'Auto-Explore' : 'Local File'}</span>
                      </p>
                    </div>
                    <div className="navigation-controls">
                      <button id="prev-btn" className="nav-btn" onClick={handlePrev}>◀ Prev</button>
                      <button id="next-btn" className="nav-btn" onClick={handleNext}>Next ▶</button>
                    </div>
                  </header>

                  <BodeCharts
                    sampleData={selectedSampleData}
                    nominalData={nominalData}
                    isAnomalous={selectedRow?.label === 'anomalous'}
                  />

                  <ParameterGrid
                    selectedRow={selectedRow}
                    headers={manifestHeaders}
                    metadata={datasetMetadata}
                  />
                </main>
              </>
            ) : activeTab === 'stats' ? (
              <main className="main-content" style={{ width: '100%' }}>
                <StatsDashboard
                  rows={manifestRows}
                  componentsList={componentsList}
                  metadata={datasetMetadata}
                />
              </main>
            ) : (
              <main className="main-content" style={{ width: '100%' }}>
                <ExperimentExplorer activeDatasetName={activeDatasetName} />
              </main>
            )}
          </div>
        </div>
      )}
    </>
  );
}

const styles: Record<string, React.CSSProperties> = {
  tabBar: {
    display: 'flex',
    padding: '4px',
    gap: '6px',
    borderRadius: 'var(--border-radius-md)',
  },
  tabBtn: {
    background: 'transparent',
    border: 'none',
    color: 'var(--color-text-muted)',
    padding: '8px 16px',
    fontFamily: 'var(--font-header)',
    fontSize: '13px',
    fontWeight: 600,
    borderRadius: 'var(--border-radius-sm)',
    cursor: 'pointer',
    transition: 'var(--transition-smooth)',
    outline: 'none',
  },
  tabBtnActive: {
    background: 'rgba(255, 255, 255, 0.08)',
    color: 'var(--color-text-main)',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)',
  },
  appLayout: {
    display: 'flex',
    flexDirection: 'column',
    width: '100vw',
    height: '100vh',
    padding: '24px',
    gap: '20px',
  },
  topNavbar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px 24px',
    borderRadius: 'var(--border-radius-lg)',
    width: '100%',
    height: '70px',
    flexShrink: 0,
  },
  navLogo: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  navLogoIcon: {
    fontSize: '22px',
  },
  navLogoText: {
    display: 'flex',
    flexDirection: 'column',
  },
  navLogoTitle: {
    fontFamily: 'var(--font-header)',
    fontSize: '18px',
    fontWeight: 800,
    letterSpacing: '-0.5px',
    lineHeight: '1.2',
  },
  navLogoSubtitle: {
    fontSize: '9px',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    color: 'var(--color-text-muted)',
    fontWeight: 600,
  },
  mainContainer: {
    display: 'flex',
    flexGrow: 1,
    gap: '24px',
    minHeight: 0,
    width: '100%',
  },
  globalDatasetContainer: {
    display: 'flex',
    alignItems: 'center',
    padding: '4px 12px',
    borderRadius: 'var(--border-radius-md)',
    gap: '8px',
  },
  globalDatasetLabel: {
    fontFamily: 'var(--font-header)',
    fontSize: '11px',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    color: 'var(--color-text-muted)',
  },
  globalDatasetSelect: {
    background: '#111625',
    border: '1px solid var(--border-glass)',
    borderRadius: 'var(--border-radius-sm)',
    color: 'var(--color-text-main)',
    fontFamily: 'var(--font-body)',
    fontSize: '13px',
    fontWeight: 600,
    cursor: 'pointer',
    outline: 'none',
    padding: '6px 12px',
    appearance: 'auto',
  },
};
