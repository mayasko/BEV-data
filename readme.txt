# BEV energy dynamics dataset

This repository contains scripts and datasets associated with the paper:

 **Unveiling Energy Dynamics of Battery Electric Vehicles Using High-Resolution Data**  
Authors: M. Yasko, A. Moussa, F. Tian, H. Kazmi, J. Driesen, and W. Martinez  
Institution: KU Leuven / EnergyVille, Thor Park 8310, Genk, Belgium  
Last updated: 2025-06-12  
Contact: mohamed.yasko@kuleuven.be

---

- `data_illustration/` — Python scripts for generating figures and illustrative plots used in the publication
- `data_processing/` — Python scripts for processing and structuring raw or pre-processed EV data
- `pylibs/` — A reference list of all Python libraries used in the project

---

## How to Use

1. Place raw `.mf4` files in a directory of your choice.
2. Use `data_processing/` scripts to decode and format the data using the provided `CAN.dbc`.
3. Analyze the structured `.csv` files by session type using the scripts in `data_illustration/`.

---

