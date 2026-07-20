import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');

// Helper to list discovered datasets in the parent data directory
function getDatasets() {
  const datasets: any[] = [];
  const seen = new Set<string>();
  
  const dataDir = path.join(REPO_ROOT, 'data');
  if (fs.existsSync(dataDir)) {
    const converters = fs.readdirSync(dataDir);
    for (const conv of converters) {
      const convPath = path.join(dataDir, conv);
      if (!fs.statSync(convPath).isDirectory()) continue;
      
      const convDataDir = path.join(convPath, `${conv}_data`);
      if (fs.existsSync(convDataDir) && fs.statSync(convDataDir).isDirectory()) {
        const folders = fs.readdirSync(convDataDir);
        for (const folder of folders) {
          const folderPath = path.join(convDataDir, folder);
          if (fs.statSync(folderPath).isDirectory()) {
            const files = fs.readdirSync(folderPath);
            const hasManifest = files.some(f => f.toLowerCase().startsWith('manifest') && f.toLowerCase().endsWith('.csv'));
            if (hasManifest) {
              const relPath = path.relative(REPO_ROOT, folderPath);
              if (!seen.has(relPath)) {
                seen.add(relPath);
                datasets.push({
                  name: folder,
                  converter: conv,
                  path: relPath
                });
              }
            }
          }
        }
      }
      
      // Check data/* directly for manifests
      const files = fs.readdirSync(convPath);
      const hasManifest = files.some(f => f.toLowerCase().startsWith('manifest') && f.toLowerCase().endsWith('.csv'));
      if (hasManifest && (fs.existsSync(path.join(convPath, 'txts')) || fs.existsSync(path.join(convPath, 'dataset.json')))) {
        const relPath = path.relative(REPO_ROOT, convPath);
        if (!seen.has(relPath)) {
          seen.add(relPath);
          datasets.push({
            name: conv,
            converter: conv,
            path: relPath
          });
        }
      }
    }
  }
  return datasets.sort((a, b) => a.converter.localeCompare(b.converter) || a.name.localeCompare(b.name));
}

// Helper to parse manifest file on request
function getManifest(datasetPath: string) {
  const fullPath = path.resolve(REPO_ROOT, datasetPath);
  if (!fs.existsSync(fullPath)) return null;
  
  const files = fs.readdirSync(fullPath);
  const manifestFile = files.find(f => f === 'manifest.csv') || files.find(f => f.toLowerCase().startsWith('manifest') && f.toLowerCase().endsWith('.csv'));
  if (!manifestFile) return null;
  
  const manifestPath = path.join(fullPath, manifestFile);
  try {
    const content = fs.readFileSync(manifestPath, 'utf8');
    const lines = content.split(/\r?\n/);
    if (lines.length === 0) return null;
    
    const parseLine = (line: string) => {
      const result: string[] = [];
      let cur = '';
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') {
          if (i + 1 < line.length && line[i + 1] === '"') {
            cur += '"';
            i++; // skip next quote
          } else {
            inQuotes = !inQuotes;
          }
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
    const rows: any[] = [];
    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      const values = parseLine(lines[i]);
      if (values.length === headers.length) {
        const row: any = {};
        for (let j = 0; j < headers.length; j++) {
          row[headers[j]] = values[j];
        }
        rows.push(row);
      }
    }
    let metadata: any = null;
    const jsonPath = path.join(fullPath, 'dataset.json');
    if (fs.existsSync(jsonPath)) {
      try {
        metadata = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
      } catch (err) {
        console.error('Error parsing dataset.json:', err);
      }
    }
    return { headers, rows, metadata };
  } catch (e) {
    console.error(e);
    return null;
  }
}

// Helper to read and parse a single Bode text sweep
function getSample(datasetPath: string, filename: string) {
  const fullDatasetPath = path.resolve(REPO_ROOT, datasetPath);
  let samplePath = path.join(fullDatasetPath, 'txts', filename);
  if (!fs.existsSync(samplePath)) {
    samplePath = path.join(fullDatasetPath, filename);
  }
  if (!fs.existsSync(samplePath)) return null;
  
  try {
    const content = fs.readFileSync(samplePath, 'utf8');
    const lines = content.split(/\r?\n/);
    const frequencies: number[] = [];
    const amplitudes: number[] = [];
    const phases: number[] = [];
    
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
  } catch (e) {
    console.error(e);
    return null;
  }
}

// Helper to list all experiment runs and their models
function getExperiments() {
  const experiments: any[] = [];
  const expDir = path.join(REPO_ROOT, 'experiments');
  if (!fs.existsSync(expDir)) return experiments;

  const runs = fs.readdirSync(expDir);
  for (const run of runs) {
    const runPath = path.join(expDir, run);
    if (!fs.statSync(runPath).isDirectory()) continue;

    const models: any[] = [];
    const modelFolders = fs.readdirSync(runPath);
    for (const m of modelFolders) {
      const modelPath = path.join(runPath, m);
      if (!fs.statSync(modelPath).isDirectory()) continue;

      const configPath = path.join(modelPath, 'model_config.json');
      const metricsPath = path.join(modelPath, 'metrics.json');
      if (fs.existsSync(configPath) && fs.existsSync(metricsPath)) {
        try {
          const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
          const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
          models.push({
            name: m,
            config,
            metrics
          });
        } catch (e) {
          console.error(`Error reading model data for ${m}:`, e);
        }
      }
    }
    
    if (models.length > 0) {
      models.sort((a, b) => a.name.localeCompare(b.name));
      experiments.push({
        run: run,
        models
      });
    }
  }
  return experiments.sort((a, b) => b.run.localeCompare(a.run));
}

// Helper to get detailed metrics, config and training history for a model run
function getExperimentDetails(runName: string, modelName: string) {
  const modelDir = path.join(REPO_ROOT, 'experiments', runName, modelName);
  if (!fs.existsSync(modelDir)) return null;

  const configPath = path.join(modelDir, 'model_config.json');
  const metricsPath = path.join(modelDir, 'metrics.json');
  const thresholdPath = path.join(modelDir, 'threshold.json');
  const historyPath = path.join(modelDir, 'training_history.json');
  const resultsPath = path.join(modelDir, 'test_results.csv');

  const result: any = {
    run: runName,
    model: modelName,
    config: null,
    metrics: null,
    threshold: null,
    history: null,
    results: null
  };

  try {
    if (fs.existsSync(configPath)) {
      result.config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    }
    if (fs.existsSync(metricsPath)) {
      result.metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
    }
    if (fs.existsSync(thresholdPath)) {
      result.threshold = JSON.parse(fs.readFileSync(thresholdPath, 'utf8'));
    }
    if (fs.existsSync(historyPath)) {
      result.history = JSON.parse(fs.readFileSync(historyPath, 'utf8'));
    }
    if (fs.existsSync(resultsPath)) {
      const content = fs.readFileSync(resultsPath, 'utf8');
      const lines = content.split(/\r?\n/);
      if (lines.length > 1) {
        const headers = lines[0].split(',');
        const rows: any[] = [];
        for (let i = 1; i < lines.length; i++) {
          const line = lines[i].trim();
          if (!line) continue;
          
          const parts: string[] = [];
          let cur = '';
          let inQuotes = false;
          for (let j = 0; j < line.length; j++) {
            const char = line[j];
            if (char === '"') {
              if (j + 1 < line.length && line[j + 1] === '"') {
                cur += '"';
                j++; // skip next quote
              } else {
                inQuotes = !inQuotes;
              }
            } else if (char === ',' && !inQuotes) {
              parts.push(cur.trim());
              cur = '';
            } else {
              cur += char;
            }
          }
          parts.push(cur.trim());
          
          if (parts.length >= headers.length) {
            const row: any = {};
            for (let j = 0; j < headers.length; j++) {
              let val: any = parts[j];
              if (typeof val === 'string') {
                if (val.startsWith('"') && val.endsWith('"')) {
                  val = val.substring(1, val.length - 1);
                }
                val = val.replace(/""/g, '"');
              }
              if (val !== '' && !isNaN(val as any)) {
                row[headers[j]] = parseFloat(val);
              } else {
                row[headers[j]] = val;
              }
            }
            rows.push(row);
          }
        }
        result.results = rows;
      }
    }
    return result;
  } catch (e) {
    console.error('Error getting experiment details:', e);
    return null;
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'api-middleware',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const url = new URL(req.url || '', 'http://localhost');
          
          if (url.pathname === '/api/datasets') {
            const datasets = getDatasets();
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify(datasets));
            return;
          }
          
          if (url.pathname === '/api/manifest') {
            const datasetPath = url.searchParams.get('path') || '';
            const manifestData = getManifest(datasetPath);
            if (!manifestData) {
              res.statusCode = 404;
              res.end('Manifest not found');
              return;
            }
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify(manifestData));
            return;
          }
          
          if (url.pathname === '/api/sample') {
            const datasetPath = url.searchParams.get('path') || '';
            const filename = url.searchParams.get('filename') || '';
            const sampleData = getSample(datasetPath, filename);
            if (!sampleData) {
              res.statusCode = 404;
              res.end('Sample not found');
              return;
            }
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify(sampleData));
            return;
          }

          if (url.pathname === '/api/experiments') {
            const experiments = getExperiments();
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify(experiments));
            return;
          }

          if (url.pathname === '/api/experiment-details') {
            const run = url.searchParams.get('run') || '';
            const model = url.searchParams.get('model') || '';
            const details = getExperimentDetails(run, model);
            if (!details) {
              res.statusCode = 404;
              res.end('Experiment details not found');
              return;
            }
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify(details));
            return;
          }
          
          next();
        });
      }
    }
  ]
});
