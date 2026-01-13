# Cosmic ray measurement

This repository contains the data analysis software for cosmic ray measurement for [Belle Integral](https://belle.kek.jp/b2j/belle-integral/).

If participant is familiar with basic [python](https://www.python.org/) and [git](https://git-scm.com/), it is encouraged that the participant ```git clone``` this repository to their local environment and process the data locally.

A simpler way to process the data is through [binder](https://mybinder.org/). One needs to open [binder](https://mybinder.org/) and paste the link to this repository ```https://github.com/sjwang03/b2integral_cosmic_ray``` on the [binder](https://mybinder.org/) homepage, then press ```lauch```. A [Jupyter](https://jupyter.org/) lab environment of this repository will be launched in the browser, where the participants can perform data analysis.

## Repositroy structure

```data/``` contains the measurement data from the oscilloscope. The output data of the analysis is also expected to be saved here.

```measurement/``` contains the scripts to acquire data from the oscilloscope.

```scripts/``` contains the scripts necessary for data analysis. An educational demonstration notebook ```demo.ipynb``` is placed here for reference ```cosmicray.py``` defines a package dedicated for this analysis, containing several helper functions which would simplify the analysis.

## Cosmicray package

Several helper functions are provided:

```decode_isf_to_csv(isf_filename: str, TIME_OFFSET: float, baseline_points: int = 1000, saturation_threshold: int = 32700)``` accepts a single ```.isf``` data file and decodes it, translating binary data into ```.csv``` file. It returns ```(waveform_amplitude, first_fwhm_timing, saturation_flag, csv_file_path)```.

```plot_waveforms_csv(csv_files, labels=None, title=None, xlim=None, ylim=None, voltage_unit="V", time_unit="ns", savepath=None, show=True)``` accepts a list of translated ```.csv``` files and make proper plots to inspect the measured waveforms.

```process_data( dir_data: str, n_measurements: int, TIME_OFFSET: float, calibrations: List[float] = [0, 0, 0, 0], summary_name: str = "summary.csv")``` accepts data directory, number of measurements to process and calibration information to process data massively, returning a path to the summary file, including waveform amplitudes, FWHM timing, calibrated FWHM timing and CSV files for all 4 channels in each measurement.