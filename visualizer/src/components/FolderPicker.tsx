import React, { useRef } from 'react';
import { FolderOpen } from 'lucide-react';

interface FolderPickerProps {
  onDirectorySelected: (files: FileList) => void;
}

export const FolderPicker: React.FC<FolderPickerProps> = ({ onDirectorySelected }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onDirectorySelected(e.target.files);
    }
  };

  return (
    <div style={styles.overlay}>
      <div className="glass" style={styles.card}>
        <FolderOpen size={48} color="#6366f1" style={{ marginBottom: 8 }} />
        <h2 style={styles.title}>Select Dataset Directory</h2>
        <p style={styles.desc}>
          To explore converter Bode sweeps, please select your dataset directory (e.g. <code>dataset_01</code> under <code>data/buck/buck_data/</code>).
        </p>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleChange}
          style={{ display: 'none' }}
          {...{
            webkitdirectory: "",
            directory: "",
            multiple: true
          } as any}
        />
        <button
          className="picker-btn"
          onClick={() => fileInputRef.current?.click()}
          style={styles.btn}
        >
          Browse Directory
        </button>
        <p style={styles.securityNote}>
          Security Note: Everything runs entirely in your local browser. No data is uploaded to any server.
        </p>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100vw',
    height: '100vh',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 100,
    background: 'rgba(8, 11, 22, 0.85)',
    backdropFilter: 'blur(8px)',
  },
  card: {
    width: '500px',
    padding: '40px',
    textAlign: 'center',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '24px',
  },
  title: {
    fontFamily: 'var(--font-header)',
    fontSize: '24px',
    fontWeight: 700,
  },
  desc: {
    color: 'var(--color-text-muted)',
    fontSize: '13px',
    lineHeight: 1.6,
  },
  btn: {
    background: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)',
    border: 'none',
    color: 'white',
    padding: '14px 28px',
    borderRadius: 'var(--border-radius-md)',
    fontFamily: 'var(--font-body)',
    fontSize: '14px',
    fontWeight: 600,
    cursor: 'pointer',
    boxShadow: '0 4px 15px rgba(79, 70, 229, 0.4)',
    outline: 'none',
    transition: 'var(--transition-smooth)',
  },
  securityNote: {
    fontSize: '11px',
    color: 'var(--color-text-muted)',
    marginTop: '-10px',
  }
};
