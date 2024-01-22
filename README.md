# GammaSpec
Data Analysis for Gamma Spectroscopy Data

# Nuclear Data Analysis Project

This project involves the analysis of nuclear data using Python. The primary goals include reading data from CSV files, 
curve fitting, error analysis, and matching measured data with the NuDat3 database. Additionally, the project identifies
background radiation and reports unidentified measured data.

## Project Structure

The project is organized into several functions and includes a main script for data processing and analysis.

### Files

- `main_script.py`: The main script that includes the functions for processing and analyzing nuclear data.
- `background_gamma.csv`: CSV file containing background gamma radiation data.
- `AGa6000_4th_attempt.csv`: CSV file containing measured nuclear data.

### Functions

1. **find_match**
   - Matches two datasets with a specified threshold value.

2. **make_float**
   - Converts a list of values to float.

3. **convert_to_float**
   - Checks and converts a value to float if possible.

4. **read_csv_column**
   - Reads a specific column from a CSV file.

5. **process_file**
   - Processes a NuDat3 CSV file, extracting energy and uncertainty values.

6. **extract_centroids**
   - Extracts centroids information from RadWare data.

7. **linear_function**
   - Defines a linear function for curve fitting.

8. **measured_energy**
   - Reads the measured data and obtains energy values.

9. **measured_cent**
   - Reads the measured data and obtains centroids.

10. **bulbul**
    - Finds matched data for provided datasets.

11. **Write_info_element_wise**
    - Prints findings in a readable format for each element/isotope.

### Fitting Part

- Linear fit is performed using the `curve_fit` function from `scipy.optimize`.

## Usage

1. Install the required Python libraries using:
   ```bash
   pip install pandas scipy matplotlib numpy

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/KaanBaspinar00/GammaSpec.git
    ```

2. Navigate to the project directory:

    ```bash
    cd GammaSpec
    ```

3. Run the program:

    ```bash
    python in_one_py.py
    ```

