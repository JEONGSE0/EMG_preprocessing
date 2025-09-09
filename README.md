# EMG_preprocessing

### EMG Signal 전처리 코드

코드 참고 🙏\
: https://github.com/pulp-bio/emgkit.git


    ap.add_argument('--in', dest='in_path', required=True, help='Input CSV path')
    ap.add_argument('--out', dest='out_path', default=None, help='Output CSV path (default: *_act.csv next to input)')
    ap.add_argument('--fs', type=float, default=250.0, help='Sampling rate in Hz (default: 250)')
    ap.add_argument('--band', nargs=2, type=float, metavar=('LOW', 'HIGH'), default=[20.0, 100.0], help='Band-pass in Hz')
    ap.add_argument('--env-cutoff', type=float, default=10.0, help='Envelope LPF cutoff (Hz), default 10')
    ap.add_argument('--detrend', choices=['none', 'mean', 'median'], default='median', help='Baseline removal (default: median)')
    ap.add_argument('--names', nargs=8, default=None, help='Explicit names of 8 EMG columns (override auto-detect)')
    ap.add_argument('--timecol', default=None, help='Optional time column to keep')
    ap.add_argument('--trim-sec', type=float, default=0.5, help='Trim first N seconds in outputs & plots')
    ap.add_argument('--view-start', type=float, default=10.0, help='Zoom preview start time (s)')
    ap.add_argument('--view-dur', type=float, default=2.0, help='Zoom preview duration (s)')
    ap.add_argument('--plot', action='store_true', help='Also show interactive plot (PNGs saved regardless)')
    ap.add_argument('--plot-pct', type=float, default=98.0, help='Percentile for autoscaling y-limits per channel (e.g., 95–99.5).')
    ap.add_argument('--plot-skip-sec', type=float, default=0.5, help='Ignore first N seconds when autoscaling (to avoid transients).')
    ap.add_argument('--plot-mode', choices=['env', 'bp', 'both'], default='env', help='What to draw in previews: envelope, bandpassed, or both.')
    ap.add_argument('--plot-norm', choices=['zscore', 'none'], default='zscore',help='Per-channel normalization for plotting only (does not affect CSV).')


