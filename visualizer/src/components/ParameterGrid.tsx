import React from 'react';

interface ParameterGridProps {
  selectedRow: Record<string, string> | null;
  headers: string[];
  metadata: any;
}

export const ParameterGrid: React.FC<ParameterGridProps> = ({ selectedRow, headers, metadata }) => {
  if (!selectedRow) {
    return (
      <section className="parameters-section glass">
        <div className="section-header">
          <h3>Component Multipliers</h3>
          <p className="section-desc">Nominal value is 1.0 (e.g. 1.05 = +5% deviation)</p>
        </div>
        <div id="parameters-grid" className="parameters-grid">
          <div className="empty-state">Select a sample to view component values</div>
        </div>
      </section>
    );
  }

  const compCols = headers.filter(
    (h) => !['filename', 'set', 'label', 'n_faults', 'mode', 'key'].includes(h)
  );

  return (
    <section className="parameters-section glass">
      <div className="section-header">
        <h3>Component Multipliers</h3>
        <p className="section-desc">Nominal value is 1.0 (e.g. 1.05 = +5% deviation)</p>
      </div>
      <div id="parameters-grid" className="parameters-grid">
        {compCols.length === 0 ? (
          <div className="empty-state">No component multipliers found in dataset</div>
        ) : (
          compCols.map((col) => {
            const val = parseFloat(selectedRow[col]);
            if (isNaN(val)) return null;

            const ranges = metadata?.component_ranges?.[col];
            
            // Check if this specific component is anomalous
            let isAnom = false;
            if (ranges) {
              if (ranges.normal && Array.isArray(ranges.normal)) {
                const [minNormal, maxNormal] = ranges.normal;
                if (val < minNormal || val > maxNormal) {
                  isAnom = true;
                }
              }
              if (ranges.anomalous && Array.isArray(ranges.anomalous)) {
                const [minAnom, maxAnom] = ranges.anomalous;
                if (val >= minAnom && val <= maxAnom) {
                  isAnom = true;
                }
              }
            } else {
              isAnom = Math.abs(val - 1.0) > 0.05;
            }

            const pctDev = (val - 1.0) * 100;
            let devClass = 'deviation-zero';
            let devText = '0.0%';

            if (Math.abs(pctDev) >= 0.01) {
              if (pctDev > 0) {
                devClass = 'deviation-up';
                devText = `+${pctDev.toFixed(1)}%`;
              } else {
                devClass = 'deviation-down';
                devText = `${pctDev.toFixed(1)}%`;
              }
            }

            return (
              <div key={col} className={`param-card ${isAnom ? 'border-red-glow' : ''}`}>
                <span className="param-name">{col}</span>
                <span className="param-val" style={isAnom ? { color: 'var(--color-red)' } : {}}>{val.toFixed(3)}</span>
                <span className={`param-deviation ${devClass}`}>{devText}</span>
                {ranges && ranges.normal && (
                  <span style={{ fontSize: '9px', color: 'var(--color-text-muted)', marginTop: '4px' }}>
                    Norm: [{ranges.normal.join(' - ')}]
                  </span>
                )}
              </div>
            );
          })
        )}
      </div>
    </section>
  );
};
