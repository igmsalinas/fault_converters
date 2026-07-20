import React, { useState, useEffect } from 'react';
import { Search } from 'lucide-react';

interface SidebarProps {
  stats: {
    total: number;
    normal: number;
    anomalous: number;
  };
  filter: string;
  setFilter: (filter: string) => void;
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  samples: Array<Record<string, string>>;
  selectedFilename: string;
  onSelectSample: (filename: string) => void;
  onChangeDirectory: () => void;
  componentsList?: string[];
  selectedComponentFilter?: string;
  onSelectComponentFilter?: (comp: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  stats,
  filter,
  setFilter,
  searchQuery,
  setSearchQuery,
  samples,
  selectedFilename,
  onSelectSample,
  onChangeDirectory,
  componentsList,
  selectedComponentFilter,
  onSelectComponentFilter
}) => {
  const [visibleLimit, setVisibleLimit] = useState(100);

  // Reset pagination limit when search query or filter changes
  useEffect(() => {
    setVisibleLimit(100);
  }, [filter, searchQuery]);

  // Load more items as the user scrolls down
  const handleScroll = (e: React.UIEvent<HTMLUListElement>) => {
    const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
    if (scrollHeight - scrollTop - clientHeight < 50) {
      setVisibleLimit((prev) => Math.min(prev + 100, samples.length));
    }
  };

  // Ensure the selected sample is always rendered in the list
  const selectedIndex = samples.findIndex((r) => r.filename === selectedFilename);
  const currentLimit = Math.max(visibleLimit, selectedIndex + 1);
  const visibleSamples = samples.slice(0, currentLimit);

  return (
    <aside className="sidebar glass">
      {/* Stats Widget */}



      {/* Stats Widget */}
      <div className="stats-widget glass-inner">
        <div className="stat-item">
          <span className="stat-val">{stats.total}</span>
          <span className="stat-lbl">Total Samples</span>
        </div>
        <div className="stat-item">
          <span className="stat-val text-green">{stats.normal}</span>
          <span className="stat-lbl">Normal</span>
        </div>
        <div className="stat-item">
          <span className="stat-val text-red">{stats.anomalous}</span>
          <span className="stat-lbl">Anomalous</span>
        </div>
      </div>

      <div className="sidebar-section">
        <label className="section-label">Filters</label>
        <div className="filter-group">
          <button
            className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
            onClick={() => setFilter('all')}
          >
            All
          </button>
          <button
            className={`filter-btn text-green ${filter === 'normal' ? 'active' : ''}`}
            onClick={() => setFilter('normal')}
          >
            Normal
          </button>
          <button
            className={`filter-btn text-red ${filter === 'anomalous' ? 'active' : ''}`}
            onClick={() => setFilter('anomalous')}
          >
            Anomalous
          </button>
        </div>
      </div>

      {/* Component Anomaly Filter (only shown when Anomalous label filter is active) */}
      {filter === 'anomalous' && componentsList && componentsList.length > 0 && onSelectComponentFilter && (
        <div className="sidebar-section">
          <label className="section-label">Filter by Anomaly Component</label>
          <div className="select-container">
            <select
              value={selectedComponentFilter}
              onChange={(e) => onSelectComponentFilter(e.target.value)}
            >
              <option value="">All Components</option>
              {componentsList.map((comp) => (
                <option key={comp} value={comp}>
                  {comp}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      <div className="sidebar-section search-section">
        <input
          type="text"
          id="search-input"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search by ID (e.g. lhs_000042)..."
          autoComplete="off"
        />
        <span className="search-icon">
          <Search size={12} />
        </span>
      </div>

      <div className="samples-list-container">
        <label className="section-label">Samples ({samples.length})</label>
        <ul id="samples-list" className="samples-list" onScroll={handleScroll}>
          {samples.length === 0 ? (
            <div className="empty-state">No matching samples</div>
          ) : (
            visibleSamples.map((row) => (
              <li
                key={row.filename}
                className={selectedFilename === row.filename ? 'selected' : ''}
                onClick={() => onSelectSample(row.filename)}
              >
                <span className="list-item-id">{row.filename.replace('.txt', '')}</span>
                <span
                  className={`list-item-badge ${
                    row.label === 'normal' ? 'badge-normal' : 'badge-anomalous'
                  }`}
                >
                  {row.label === 'normal' ? 'Normal' : 'Anom'}
                </span>
              </li>
            ))
          )}
        </ul>
      </div>

      <button
        className="nav-btn"
        style={{ width: '100%', borderColor: 'rgba(255,255,255,0.15)', marginTop: 'auto' }}
        onClick={onChangeDirectory}
      >
        Change Directory
      </button>
    </aside>
  );
};
