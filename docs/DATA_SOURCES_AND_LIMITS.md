# Data Sources And Limits

## Primary Dataset

The final prototype uses local Aditya-L1 Level-2 data already present in this project:

- SWIS TH1 CDF files in `tha1/`
- SWIS TH2 CDF files in `tha2/`
- SWIS BLK CDF files in `swis_BLK/`
- MAG L2 NetCDF files in `mag_2026Aug23T210145602/`

The final scientific build uses `2024-08-09` through `2024-08-15`.

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

## Other Possible Context Data

Public Wind, DSCOVR, and OMNI data are available through NASA/SPDF, NOAA/NCEI, and OMNIWeb. These can be useful for future comparison, but they should not be rushed into the final submission unless they are carefully time-aligned and documented.

Useful sources:

- NASA/SPDF OMNIWeb: https://omniweb.gsfc.nasa.gov/
- NASA CDAWeb: https://cdaweb.gsfc.nasa.gov/
- NOAA DSCOVR/NCEI: https://www.ncei.noaa.gov/products/deep-space-climate-observatory-dscovr

## Final Submission Position

For tomorrow's submission, the safest position is:

- Use the Aditya-L1 SWIS/MAG August 2024 dataset as the main result.
- Use Zenodo November 2023 only as a secondary SWIS-only OPDI portability demo.
- Mention PRADAN as the official route for adding more Aditya-L1 mission data.
- Keep scientific claims limited to a single-event exploratory prototype.
