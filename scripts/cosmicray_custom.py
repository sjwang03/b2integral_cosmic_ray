import os
import re
import struct
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from scipy.optimize import curve_fit

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def decode_isf_to_csv(isf_filename: str,TIME_OFFSET: float, ratio_time = True, ratio_threshold = 0.5, V_threshold = -0.05,
                      baseline_points: int = 1000,
                      saturation_threshold: int = 32700):
    """
    Decode a Tektronix-like .isf file:
      - Parse XINCR (s) and YMULT (V) from header
      - Find ':CURVE #<N><bytecount>' binary block
      - Decode 2-byte signed integers (endianness from BYT_OR)
      - Write CSV with columns: time_ns, voltage_V
      - TIME_OFFSET is in ns (subtracted from time axis)
      - If ratio_time is True, the timing threshold is determined by ratio of pulse height

    Returns:
      (vmin_V, signal_timing_ns, saturation_flag, csv_path)
    """

    isf_path = Path(isf_filename)
    csv_path = isf_path.with_suffix(".csv")

    with open(isf_path, "rb") as f:
        data = f.read()

    # --- 1) Locate CURVE block start: ':CURVE #'
    marker = b":CURVE #"
    idx = data.find(marker)
    if idx < 0:
        raise ValueError("Cannot find ':CURVE #' marker in file.")

    # Header bytes are everything before ':CURVE #'
    header_bytes = data[:idx]
    # Tek headers are ASCII-ish; latin-1 avoids decode errors on odd bytes.
    header = header_bytes.decode("latin-1", errors="replace")

    # --- 2) Parse key fields from header
    def _get_float(key: str) -> float:
        # Matches patterns like 'XINCR 2.0E-10' or 'YMULT 7.8125E-5'
        m = re.search(rf"\b{re.escape(key)}\s+([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\b", header)
        if not m:
            raise ValueError(f"Cannot find {key} in header.")
        return float(m.group(1))

    def _get_str(key: str) -> str:
        m = re.search(rf"\b{re.escape(key)}\s+([A-Za-z0-9_]+)\b", header)
        if not m:
            raise ValueError(f"Cannot find {key} in header.")
        return m.group(1)

    XINCR_s = _get_float("XINCR")      # seconds per sample
    YMULT = _get_float("YMULT")        # volts per count (scale)
    XINCR_ns = XINCR_s * 1e9

    # Endianness: BYT_OR MSB -> big-endian, LSB -> little-endian
    byt_or = _get_str("BYT_OR")
    endian = ">" if byt_or.upper() == "MSB" else "<"

    # Optional: number of points (NR_PT) if present
    nr_pt = None
    try:
        nr_pt = int(_get_float("NR_PT"))
    except Exception:
        pass

    # --- 3) Parse the binary block length after ':CURVE #'
    # Format: ':CURVE #'<N><bytecount><binary...>
    # Example: ':CURVE #520000' -> N=5, bytecount=20000
    p = idx + len(marker) # place the pointer
    if p >= len(data):
        raise ValueError("File ends unexpectedly after ':CURVE #'.")

    n_digits_char = data[p:p+1] # <N>
    if not n_digits_char.isdigit():
        raise ValueError("Invalid CURVE block: expected digit after '#'.")
    n_digits = int(n_digits_char)
    p += 1

    bytecount_bytes = data[p:p+n_digits] # <bytecount>
    if len(bytecount_bytes) != n_digits or not bytecount_bytes.isdigit():
        raise ValueError("Invalid CURVE block: cannot read bytecount.")
    bytecount = int(bytecount_bytes)
    p += n_digits

    binary = data[p:p+bytecount]
    if len(binary) != bytecount:
        raise ValueError(f"Binary block truncated: expected {bytecount} bytes, got {len(binary)}.")

    # Expect 2 bytes per sample
    if bytecount % 2 != 0:
        raise ValueError(f"Binary bytecount {bytecount} is not divisible by 2 (expected 2-byte integers).")

    n_samples = bytecount // 2
    if nr_pt is not None and nr_pt != n_samples:
        # Not fatal, but good to know
        print(f"NR PT in metadata: {nr_pt} while samples in bytecount: {n_samples}.")
        pass

    # --- 4) Decode int16 array with correct endianness
    # numpy can do this efficiently:
    dtype = np.dtype(endian + "i2")  # signed 16-bit
    v_raw = np.frombuffer(binary, dtype=dtype).astype(np.float64)

    # Saturation check 
    saturation = int(np.any((v_raw > saturation_threshold) | (v_raw < -saturation_threshold)))

    # Baseline subtraction using first baseline_points samples
    n_bl = min(baseline_points, len(v_raw)) # v_raw should be 10k 
    offset = float(np.mean(v_raw[:n_bl])) if n_bl > 0 else 0.0

    # Scale to volts
    v_V = (v_raw - offset) * YMULT

    # Time axis in ns
    t_ns = np.arange(len(v_V), dtype=np.float64) * XINCR_ns - float(TIME_OFFSET)

    # --- 5) Pulse height + signal timing
    # find minimum point (pulse bottom)
    if len(v_V):
        imin = np.argmin(v_V)
        vmin = v_V[imin]

        # threshold

        # Exercise 0: implement the threshold calculation logic.
        # if ratio_time is true, the threshold should be calculated based on V_min and ratio_threshold, otherwise it should be the fixed value V_threshold.
        threshold = 0 # placeholder, replace with your code

        # search BACKWARD from minimum
        signal_timing = np.nan
        for i in range(imin, -1, -1):
            if v_V[i] > threshold:
                # crossing between i and i+1
                # optional linear interpolation
                t1, t2 = t_ns[i], t_ns[i+1]
                v1, v2 = v_V[i], v_V[i+1]

                frac = (threshold - v1) / (v2 - v1)
                signal_timing = t1 + frac * (t2 - t1)
                break
    else:
        raise ValueError(f"Invalid data length")

    # --- 6) Write CSV
    # Columns: time_ns, voltage_V
    with open(csv_path, "w", encoding="utf-8") as out:
        out.write("time_ns,voltage_V\n")
        for ti, vi in zip(t_ns, v_V):
            out.write(f"{ti:.6f},{vi:.9f}\n")

    return vmin, signal_timing, saturation, str(csv_path)

def plot_waveforms_csv(csv_files,
                       labels=None,
                       title=None,
                       xlim=None,
                       ylim=None,
                       voltage_unit="V",
                       time_unit="ns",
                       plot_time=True,
                       ratio_time=True,
                       ratio_threshold=0.5,
                       V_threshold=-0.05,
                       savepath=None,
                       show=True):

    csv_files = [Path(f) for f in csv_files]
    if labels is None:
        labels = [f.stem for f in csv_files]
    if len(labels) != len(csv_files):
        raise ValueError("labels length must match csv_files length")

    # --- scaling ---
    y_scale = 1.0
    y_label = "Voltage (V)"
    if voltage_unit.lower() == "mv":
        y_scale = 1e3
        y_label = "Voltage (mV)"

    unit_scale = {"ns": 1.0, "us": 1e-3, "ms": 1e-6, "s": 1e-9}
    if time_unit.lower() not in unit_scale:
        raise ValueError(f"Unsupported time_unit: {time_unit}")
    x_scale = unit_scale[time_unit.lower()]
    x_label = f"Time ({time_unit})"

    fig, ax = plt.subplots()

    for f, lab in zip(csv_files, labels):
        data = np.genfromtxt(f, delimiter=",", names=True)

        t = np.array(data["time_ns"], dtype=float)
        v = np.array(data["voltage_V"], dtype=float)

        order = np.argsort(t)
        t = t[order]
        v = v[order]

        # scaled for plotting
        tp = t * x_scale
        vp = v * y_scale

        line, = ax.plot(tp, vp, label=lab)
        color = line.get_color()


        # -----------------------------
        # Timing extraction
        # -----------------------------
        if plot_time:
            if len(v) == 0:
                continue

            imin = np.argmin(v)
            vmin = v[imin]

            # Exercise 0: implement the threshold calculation logic.
            # if ratio_time is true, the threshold should be calculated based on V_min and ratio_threshold, otherwise it should be the fixed value V_threshold.
            threshold = 0 # placeholder, replace with your code

            timing = np.nan

            for i in range(imin, -1, -1):
                if v[i] > threshold:
                    t1, t2 = t[i], t[i+1]
                    v1, v2 = v[i], v[i+1]
                    frac = (threshold - v1) / (v2 - v1)
                    timing = t1 + frac * (t2 - t1)
                    break

            if np.isfinite(timing):
                # draw lines
                ax.axhline(threshold * y_scale, linestyle="--", alpha=0.6, color=color)
                ax.axvline(timing * x_scale, linestyle="--", alpha=0.6, color=color)

                # crossing point
                ax.scatter(timing * x_scale, threshold * y_scale, s=10, color=color)
                print(f"{lab}: vmin={vmin:.4g} V, threshold={threshold:.4g} V, timing={timing:.3f} ns")

    # -----------------------------
    if title:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.legend()

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=200)

    if show:
        plt.show()

    return fig, ax

def process_data(
    dir_data: str,
    n_measurements: int,
    TIME_OFFSET: float,
    calibrations: List[float] = [0, 0, 0, 0],
    summary_name: str = "summary.csv",
    ratio_time = True, ratio_threshold = 0.5, V_threshold = -0.05
) -> str:
    """
    Process waveform data and save summary CSV.

    Output columns:
      measurement,
      v1..v4,
      t1..t4,
      ct1..ct4,
      csv1..csv4

    Returns:
      summary_csv_path
    """

    if not os.path.isdir(dir_data):
        raise FileNotFoundError(f"Directory does not exist: {dir_data}")

    summary_csv_path = os.path.join(dir_data, summary_name)

    with open(summary_csv_path, "w", encoding="utf-8") as fsum:

        # header
        fsum.write(
            "measurement,"
            "v1,v2,v3,v4,"
            "t1,t2,t3,t4,"
            "ct1,ct2,ct3,ct4,"
            "csv1,csv2,csv3,csv4\n"
        )

        for m in range(n_measurements):
            vmins = []
            signal_timings = []
            csvs = []

            for c in range(4):  # 4 channels
                isf = os.path.join(dir_data, f"run{m:05d}_{c+1}.isf")

                vmin, signal_timing, saturation, csv = decode_isf_to_csv(
                    isf, TIME_OFFSET, ratio_time, ratio_threshold,V_threshold
                )

                vmins.append(vmin)
                signal_timings.append(signal_timing)
                csvs.append(csv)

                if saturation == 1:
                    print(f"[run {m:05d} ch {c+1}] saturation?")

            # calibrated times
            # Exercise 0: apply the calibration to the signal timings. 
            ct = signal_timings # placeholder, replace with your code

            # write one row
            row = [
                m,
                vmins[0], vmins[1], vmins[2], vmins[3],
                signal_timings[0], signal_timings[1], signal_timings[2], signal_timings[3],
                ct[0], ct[1], ct[2], ct[3],
                csvs[0], csvs[1], csvs[2], csvs[3],
            ]

            fsum.write(",".join(str(x) for x in row) + "\n")

    print(f"Summary saved to: {summary_csv_path}")
    return summary_csv_path

def plot_hist_gaussfit(
    tof,
    bins=100,
    plt_range=None,
    xlabel="TOF [ns]",
    title="TOF distribution",
):
    """
    tof : pandas Series or numpy array
    bins : number of bins
    plt_range : (xmin, xmax)
    """

    # remove NaN
    tof = np.asarray(tof)
    tof = tof[np.isfinite(tof)]

    # histogram
    counts, edges = np.histogram(tof, bins=bins, range=plt_range)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_width = edges[1] - edges[0]

    # initial guess
    p0 = [
        counts.max(),         # A
        np.mean(tof),         # mu
        np.std(tof)           # sigma
    ]

    # fit
    popt, pcov = curve_fit(gauss, centers, counts, p0=p0)
    A, mu, sigma = popt

    # plot histogram
    plt.figure(figsize=(7,5))
    plt.hist(tof, bins=bins, range=plt_range, alpha=0.9, label="Data", histtype="step", linewidth=2)

    # plot fit
    xfit = np.linspace(edges[0], edges[-1], 1000)
    plt.plot(xfit, gauss(xfit, *popt), 'r-', lw=2, label="Gaussian fit")

    # labels
    plt.xlabel(xlabel)
    plt.ylabel(f"Entries / {bin_width:.3f} ns")
    plt.title(title)

    # text on plot
    text = (
        f"$\\mu$ = {mu:.3f} ns\n"
        f"$\\sigma$ = {sigma:.3f} ns\n"
        f"Entries = {len(tof)}"
    )
    plt.text(
        0.97, 0.4, text,
        transform=plt.gca().transAxes,
        ha='right', va='top',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    plt.legend()
    # plt.tight_layout()
    plt.grid(True)
    plt.show()

    # print
    print("Fit result:")
    print(f"Mean  = {mu:.6f} ns")
    print(f"Sigma = {sigma:.6f} ns")

    return mu, sigma
