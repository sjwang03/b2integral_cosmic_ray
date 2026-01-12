import os
import re
import struct
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def decode_isf_to_csv(isf_filename: str, TIME_OFFSET: float,
                      baseline_points: int = 1000,
                      saturation_threshold: int = 32700):
    """
    Decode a Tektronix-like .isf file:
      - Parse XINCR (s) and YMULT (V) from header
      - Find ':CURVE #<N><bytecount>' binary block
      - Decode 2-byte signed integers (endianness from BYT_OR)
      - Write CSV with columns: time_ns, voltage_V
      - TIME_OFFSET is in ns (subtracted from time axis)

    Returns:
      (vmin_V, first_fwhm_timing_ns, saturation_flag, csv_path)
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

    # Saturation check (like your original)
    saturation = int(np.any((v_raw > saturation_threshold) | (v_raw < -saturation_threshold)))

    # Baseline subtraction using first baseline_points samples
    n_bl = min(baseline_points, len(v_raw)) # v_raw should be 10k 
    offset = float(np.mean(v_raw[:n_bl])) if n_bl > 0 else 0.0

    # Scale to volts (match your old logic: (raw - offset) * YMULT)
    v_V = (v_raw - offset) * YMULT

    # Time axis in ns
    t_ns = np.arange(len(v_V), dtype=np.float64) * XINCR_ns - float(TIME_OFFSET)

    # --- 5) Pulse height + first 50% crossing (your old outputs)
    # find minimum point (pulse bottom)
    if len(v_V):
        imin = np.argmin(v_V)
        vmin = v_V[imin]

        # 50% threshold
        threshold = 0.5 * vmin   # vmin is negative

        # search BACKWARD from minimum
        FWHM_timing = np.nan
        for i in range(imin, -1, -1):
            if v_V[i] > threshold:
                # crossing between i and i+1
                # optional linear interpolation
                t1, t2 = t_ns[i], t_ns[i+1]
                v1, v2 = v_V[i], v_V[i+1]

                frac = (threshold - v1) / (v2 - v1)
                FWHM_timing = t1 + frac * (t2 - t1)
                break
    else:
        raise ValueError(f"Invalid data length")

    # --- 6) Write CSV
    # Columns: time_ns, voltage_V
    with open(csv_path, "w", encoding="utf-8") as out:
        out.write("time_ns,voltage_V\n")
        for ti, vi in zip(t_ns, v_V):
            out.write(f"{ti:.6f},{vi:.9f}\n")

    return vmin, FWHM_timing, saturation, str(csv_path)

def plot_waveforms_csv(csv_files,
                       labels=None,
                       title=None,
                       xlim=None,
                       ylim=None,
                       voltage_unit="V",   # "V" or "mV"
                       time_unit="ns",     # "ns" or "us" etc (only affects scaling if you choose)
                       savepath=None,
                       show=True):
    """
    Plot multiple waveforms from CSVs with columns: time_ns, voltage_V

    Args:
      csv_files: list[str | Path]
      labels: list[str] or None (defaults to stem of filename)
      title: plot title
      xlim, ylim: tuple(min,max) or None
      voltage_unit: "V" or "mV" (scales Y axis)
      time_unit: currently expects input in ns; if you set "us" it will scale by 1e-3, etc.
      savepath: if set, saves figure to this path (e.g. "waveforms.png")
      show: if True, plt.show()

    Returns:
      (fig, ax)
    """
    csv_files = [Path(f) for f in csv_files]
    if labels is None:
        labels = [f.stem for f in csv_files]
    if len(labels) != len(csv_files):
        raise ValueError("labels length must match csv_files length")

    # scaling
    y_scale = 1.0
    y_label = "Voltage (V)"
    if voltage_unit.lower() == "mv":
        y_scale = 1e3
        y_label = "Voltage (mV)"

    # input CSV uses time_ns; scale if user wants different unit
    # ns -> us: 1e-3, ns -> ms: 1e-6, ns -> s: 1e-9
    unit_scale = {"ns": 1.0, "us": 1e-3, "ms": 1e-6, "s": 1e-9}
    if time_unit.lower() not in unit_scale:
        raise ValueError(f"Unsupported time_unit: {time_unit}. Use one of {list(unit_scale.keys())}")
    x_scale = unit_scale[time_unit.lower()]
    x_label = f"Time ({time_unit})"

    fig, ax = plt.subplots()

    for f, lab in zip(csv_files, labels):
        data = np.genfromtxt(f, delimiter=",", names=True)
        if "time_ns" not in data.dtype.names or "voltage_V" not in data.dtype.names:
            raise ValueError(f"{f} must have columns: time_ns, voltage_V")

        t = np.array(data["time_ns"], dtype=float) * x_scale
        v = np.array(data["voltage_V"], dtype=float) * y_scale

        # sort by time just in case
        order = np.argsort(t)
        ax.plot(t[order], v[order], label=lab)

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
            fwhms = []
            csvs = []

            for c in range(4):  # 4 channels
                isf = os.path.join(dir_data, f"run{m:05d}_{c+1}.isf")

                vmin, fwhm, saturation, csv = decode_isf_to_csv(
                    isf, TIME_OFFSET
                )

                vmins.append(vmin)
                fwhms.append(fwhm)
                csvs.append(csv)

                if saturation == 1:
                    print(f"[run {m:05d} ch {c+1}] saturation?")

            # calibrated times
            ct = [
                fwhms[i] + calibrations[i]
                for i in range(4)
            ]

            # write one row
            row = [
                m,
                vmins[0], vmins[1], vmins[2], vmins[3],
                fwhms[0], fwhms[1], fwhms[2], fwhms[3],
                ct[0], ct[1], ct[2], ct[3],
                csvs[0], csvs[1], csvs[2], csvs[3],
            ]

            fsum.write(",".join(str(x) for x in row) + "\n")

    print(f"Summary saved to: {summary_csv_path}")
    return summary_csv_path