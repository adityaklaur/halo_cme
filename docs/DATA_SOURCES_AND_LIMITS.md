# Data Sources And Limits

## Primary Datasets

The detector demonstration uses local August 2024 Aditya-L1 Level-2 data already present in this project:

- SWIS TH1 CDF files in `tha1/`
- SWIS TH2 CDF files in `tha2/`
- SWIS BLK CDF files in `swis_BLK/`
- MAG L2 NetCDF files in `mag_2026Aug23T210145602/`

The Phase 2 registry additionally uses processed Aditya-L1 sources for October 2024, September 2024 and March 2025. November 2024 is registered but intentionally fails the modality contract until valid SWIS and MAG inputs are supplied.

## Official Source For More Aditya-L1 Data

Additional official Aditya-L1 data should be downloaded from ISRO/ISSDC PRADAN:

- https://pradan.issdc.gov.in/al1
- https://pradan1.issdc.gov.in/al1

These portals may require registration/login. Because of that, the project should not pretend to automatically download the full official mission archive.

## Public External Dataset Already Added

A public Zenodo SWIS Level-2 sample was downloaded and processed during development:

- Dataset: `AL1-ASPEX-SWIS [06-12 November 2023], L2 Data`
- DOI: `10.5281/zenodo.15861770`
- GitHub-clean note: the downloaded external files were moved out of the repo to `../Local_Data_Folders/Aditya_L2_local_data_20260824/data/external/`.
- Restore path if needed: copy that folder back to `data/external/zenodo_swis_20231106_12/`.

This dataset is useful for showing that OPDI can be computed on another SWIS TH1/TH2 interval.

Important limit: this Zenodo package is SWIS-only in this project. It does not include matching BLK plasma and MAG context for full CME validation.

## NASA OMNI Context Data

Official one-minute `OMNI_HRO_1MIN` CSV exports are packaged for the four added intervals. The files provide GSE magnetic components, solar-wind speed, proton density and temperature shifted to Earth's bow-shock nose. They are quality-filtered using the published fill values and stored with `omni_` column prefixes.

OMNI is external near-Earth context. It is not an Aditya-L1 product and cannot replace missing SWIS spectra, BLK plasma or MAG observations.

Useful sources:

- NASA/SPDF OMNIWeb: https://omniweb.gsfc.nasa.gov/
- NASA CDAWeb: https://cdaweb.gsfc.nasa.gov/
- NOAA DSCOVR/NCEI: https://www.ncei.noaa.gov/products/deep-space-climate-observatory-dscovr

## Scientific Position

- Five independent source intervals are represented.
- Six of seven registered windows meet the current Aditya-L1 completeness contract.
- October 12 SWIS and partial September products are preserved as source gaps.
- November 25 remains blocked by invalid/missing Aditya-L1 inputs.
- Multi-event detector-performance claims remain disabled until the blocked window passes.
