import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import csv
import numpy as np
import re

# ALL functions:

#################################### FUNCTIONS ###########################################

# this matches 2 given datasets with the threshold value.
def find_match(experimental, nudat, threshold=1.49):
    matched_nudat = []
    matched_exp = []
    differences = []
    unidentified_measured = []
    unidentified_nudat = []

    for i in experimental:
        match_found = False
        for j in nudat:
            diff = np.abs(i - j)
            if diff <= threshold:
                matched_nudat.append(j)
                matched_exp.append(i)
                differences.append(diff)
                match_found = True
                break

        if not match_found:
            unidentified_measured.append(i)

    unidentified_nudat = [j for j in nudat if j not in matched_nudat]

    return matched_nudat, matched_exp, differences, unidentified_measured, unidentified_nudat


# Make them float using the function below
def make_float(liste):
    listem = [float(value) for value in liste]
    return listem

# Checks the data and if it is possible, turn it into float
def convert_to_float(value):
    if isinstance(value, (float, int)):
        # If the value is already a float or int, return it as is
        return value
    elif isinstance(value, str):
        try:
            # Try to convert the string to a float
            return float(value)
        except ValueError:
            # If the conversion fails, return an appropriate value or handle the error as needed
            return None
    else:
        # If the input is neither a float nor a string, handle it as needed
        return None

def read_csv_column(csv_file_path, column_name):
    """
    Read a specific column from a CSV file and return its values.

    Parameters:
    - csv_file_path (str): Path to the CSV file.
    - column_name (str): Name of the column to read.

    Returns:
    - list: Values in the specified column.
    """
    # Open the CSV file
    with open(csv_file_path, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)

        # Read the header row to get the column names
        header = next(csv_reader)
        if column_name == isinstance(column_name,int):
            # Get the index of the specified column
            column_index = header.index(column_name)
            # Check if the specified column exists in the header
            if column_name not in header:
                raise ValueError(f"Column '{column_name}' not found in the CSV file.")

        else:
            column_index = column_name

        # Initialize a list to store the values in the specified column
        column_values = []

        # Iterate through each row in the CSV file
        for row in csv_reader:
            # Append the value in the specified column to the list
            column_values.append(row[column_index])
    """
    # Example usage:
    csv_file_path = 'path/to/your/file.csv'
    column_name = 'desired_column'
    result = read_csv_column(csv_file_path, column_name)
    print(result)
    """
    return column_values


# Copy and past the data from nudat. Then give it to this function.
# It will return energy and uncertainty values.
def process_file(file_name):
    try:
        # Read the data into a DataFrame
        df = pd.read_csv(file_name, sep='\t', header=None)

        # Extract the specified column (1st column)
        specified_column = df.iloc[:, 0]

        # Convert specified column to a list
        specified_column = list(specified_column)

        # Initialize empty lists for energy and unc
        energy = []
        unc = []

        # Iterate through the input list
        for item in specified_column:
            try:
                # Split each string into parts based on space
                parts = item.split()

                # Convert the first part to a float and append to the energy list
                energy.append(float(parts[0]))

                # Convert the second part to an integer and append to the unc list
                unc.append(int(parts[1]) if len(parts) > 1 and parts[1]=="0" else 0)
            except (ValueError, IndexError):
                # Handle exceptions (e.g., if conversion fails or index is out of range)
                print(f"Skipping invalid item: {item}")

        # Return the result
        return energy, unc

    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None

# Extract centroids info from RadWare. Copy and paste data to the csv file.
# Give it to this function.

def extract_centroids(file_path):
    # Open the CSV file and read the first column
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')  # Assuming it's tab-separated, adjust delimiter as needed
        next(reader)  # Skip the header row if there is one

        # Function to remove (number) from the first column
        def remove_number_parts(line):
            return re.sub(r'(\d+\.\d+).*', r'\1', line)

        # Extract the first column after removing (number) parts
        centroids = [remove_number_parts(row[0]).strip() for row in reader]

    return centroids

### 000000000000000000000000000000000000000000000000000000000000000000000000000000000 ###
# Define the linear function
def linear_function(x, m, b):
    return m * x + b

# This function reads the data and gets the energy values.
def measured_energy(file_name_measured):
    exp_cent = make_float(list(extract_centroids(file_name_measured)))
    # extract_centroids function reads centroids.
    # make_float function turns the values which are in string format to the float

    exp_energy = linear_function(np.array(exp_cent), params[0], params[1])
    # using linear_function, we get energy values from centroids.

    return exp_energy

def measured_cent(file_name_measured):
    exp_cent = make_float(list(extract_centroids(file_name_measured)))
    exp_energy = linear_function(np.array(exp_cent), params[0], params[1])

    return exp_cent

# bulbul function finds matched data for provided dataset.
def bulbul(file_name_list_nudat,file_name_measured):
    Great_matched_list = []

    for file_name_nudat in file_name_list_nudat:
        Energy_of_element_nudat = process_file(file_name_nudat)[0]

        Uncertanity_of_element_nudat = process_file(file_name_nudat)[1]
        exp_energy= measured_energy(file_name_measured)

        Matched_element = find_match(exp_energy,Energy_of_element_nudat,threshold=1.499)

        Great_matched_list.append(Matched_element)

    return Great_matched_list

    ## x: element index in the list
    ## Great_match_list[x][0]: Nudat_matched
    ## Great_match_list[x][1]: Measurment_matched
    ## Great_match_list[x][2]: Differences_matched



# function below prints findings in a readable format.
def Write_info_element_wise(file_name_list_nudat,file_name_measured):
    for i in range(len(file_name_list_nudat)):
        NAME = file_name_list_nudat[i][:-9]

        print("\nElement/Isotope Name:",NAME)

        Nudat_matched = bulbul(file_name_list_nudat, file_name_measured)[i][0]
        Measurment_matched = bulbul(file_name_list_nudat, file_name_measured)[i][1]

        print("\nMatchs from Nudat: \n",Nudat_matched)
        print("\nMatchs from Measured: \n",Measurment_matched)

##########################################################################################

### Fitting Part:
# Data are taken from BeforeCalibration datasets which Haris Dapo provided.
# Fitting is done using these data (shown below)

data = {
    'Elements': ['Ba133', 'Ba133', 'Ba133', 'Ba133', 'Ba133', 'Ba133', 'Ba133', 'Ba133', 'Cd109', 'Co60', 'Co60', 'Co57', 'Co57', 'Cs137', 'Mn54', 'Na22', 'Na22'],
    'Centroids': [277.55, 421.951, 836.96, 1162.26, 1439.13, 1576.723, 1853.196, 1997.896, 459.14, 6100.672, 6928.44, 636.253, 711.22, 3441.898, 4342.07, 2658.68, 6627.181],
    'Unc_Centroids': [13, 11, 21, 23, 3, 16, 9, 25, 4, 22, 24, 14, 4, 11, 3, 18, 25],
    'Calibrated Energy': [53.17, 80.928, 160.69, 223.22, 276.436, 302.882, 356.0216, 383.834, 88.075, 1172.411, 1331.513, 122.118, 136.528, 661.3793, 834.398, 510.84, 1273.609],
    'Unc_energy': [3, 21, 4, 4, 5, 3, 17, 5, 8, 4, 5, 3, 8, 21, 6, 3, 5],
    'Energy from Nudat': [53.1622, 80.9979, 160.612, 223.2368, 276.3989, 302.8508, 356.0129, 383.8485, 88.0336, 1173.228, 1332.492, 122.06065, 136.47356, 661.657, 834.848, 511, 1274.537],
    'Unc_Nudat': [6, 11, 16, 13, 12, 5, 7, 12, 10, 3, 4, 12, 29, 3, 3, 0, 7],
    'Energy Differences': [0.0078, 0.0699, 0.078, 0.0168, 0.0371, 0.0312, 0.0087, 0.0145, 0.0414, 0.817, 0.979, 0.0573, 0.0544, 0.2777, 0.45, 0.16, 0.928]
}

df = pd.DataFrame(data)

print(np.array(df['Centroids']))
print(np.array(df['Energy from Nudat']))


Centroids = np.array(df['Centroids'])
Nudat_Energy= np.array(df['Energy from Nudat'])


# Perform the linear fit using curve_fit
params, covariance = curve_fit(linear_function, Centroids, Nudat_Energy)

# Extract the slope and intercept from the parameters
slope, intercept = params

# Calculate errors in slope and intercept from the covariance matrix
slope_error, intercept_error = np.sqrt(np.diag(covariance))

# Print the results
print(f"Slope: {slope:.8f} ± {slope_error:.8f}")
print(f"Intercept: {intercept:.8f} ± {intercept_error:.8f}")

# Assuming you have already performed the linear fit and obtained the covariance matrix
covariance_matrix = covariance

# Extract variances and covariances
var_a = covariance_matrix[0, 0]
var_b = covariance_matrix[1, 1]
cov_ab = covariance_matrix[0, 1]
print("covariance matrix: \n",covariance_matrix,"\ncorrelation matrix: \n",np.corrcoef(covariance_matrix))
# Calculate correlation coefficient
rho_ab = cov_ab / np.sqrt(var_a * var_b)

# Given values
a = slope  # Replace with your slope value from the linear fit
x = Centroids
sigma_x = np.array([0.13, 0.011, 0.21, 0.23, 0.03, 0.016,  0.009, 0.025,  0.04, 0.022, 0.24, 0.014,  0.04, 0.011,  0.03, 0.18, 0.025])

print("std of x: ",sigma_x)
sigma_a = slope_error  # Uncertainty in slope
sigma_b = intercept_error  # Uncertainty in intercept
print("uncertanity in slope: ",sigma_a,"\nuncertanity in intercept: ",sigma_b)
rho_ab = rho_ab  # Correlation coefficient between a and b
print("rho_ab: ",rho_ab)

# Calculate the uncertainty using the provided formula
sigma_f2 = (a**2) * (sigma_x**2) + (x**2) * (sigma_a**2) + sigma_b**2 + 2 * x * sigma_a * sigma_b * rho_ab
print(f"Uncertainty (sigma_f): \n{np.sqrt(sigma_f2)}")

fit_values = linear_function(Centroids, slope, intercept)
# Calculate residuals: the differences between observed and fitted values
residuals = Nudat_Energy - fit_values
errors = np.std(residuals)

# Calculate chi-squared: a measure of the goodness of fit
chi_squared = np.sum((residuals / errors) ** 2)
print("chi_squared = ", chi_squared)

# Calculate degrees of freedom
df = len(fit_values) - 2

# Calculate reduced chi-squared: a normalized version of chi-squared
chi_squared_r = chi_squared / df
print("dof: ",df)
print("Reduced Chi-squared: ", chi_squared_r,"\n")

# Plot the data and the linear fit
plt.scatter(Centroids, Nudat_Energy, label='Data')
plt.plot(Centroids, linear_function(Centroids, slope, intercept), color='red', label='Linear Fit')
plt.xlabel('Centroids')
plt.ylabel('Nudat Energy')
plt.grid()
plt.legend()
#plt.show()


#####################################################################################
#####################################################################################
#####################################################################################


#### Background list
# These are the possible radiations from background. (taken from 2 different articles)

BG_from_article1 = [46.5, 53.2, 63.3, 92.4, 143.8, 163.1, 185.7, 186.2, 238.6, 242.0, 295.2, 300.1, 338.3, 351.9, 463.0, 510.8, 583.2, 609.3, 661.7, 665.5, 727.3, 768.4, 785.7, 806.2, 860.6, 911.2, 934.0, 964.0, 968.0, 1001.0, 1120.0, 1155.0, 1173.0, 1238.0, 1280.0, 1332.0, 1377.0, 1385.0, 1401.0, 1407.0, 1460.0, 1508.0, 1538.0, 1543.0, 1583.0, 1660.0, 1729.0, 1764.0, 1847.0, 2117.0, 2203.0, 2447.0, 2614.0]
BG_from_article2 = [53.5, 66.6, 139.7, 198.4, 477.6, 511.0, 537.4, 558.4, 569.7, 595.9, 669.6, 691.5, 718.3, 803.1, 834.0, 843.8, 846.8, 962.1, 1014.4, 1039.2, 1368.6, 1778.8, 2223.3, 6506.0, 6783.0, 7416.0, 10199.0]
BGs = BG_from_article1+BG_from_article2
Background = list(np.array(read_csv_column("background_gamma.csv",int(0))))
Background_all = [convert_to_float(i) for i in Background]
Background_all = np.sort(list(set(Background_all+BGs)))

#print("Background_all: ",Background_all)
"""Ga_exp_cent = [3126.0, 3276.88, 4089.7, 4192.2, 4337.65,
                      4475.7, 4649.9, 5404.4, 5465.6, 5601.94, 5825.1, 6400.0, 6436.9, 6553.2, 7115.9, 7162.7,
                      8256.7, 8302.2, 8477.0, 9603.3, 9674.3, 9789.6, 10941.0, 11013.2, 11445.58, 12947.0, 13033.0,
                      14313.7]"""

# This part tries to match data taken from HPGe detector with the data from NuDat3
# You do not need to compare data with all of these. You may change if you need.

file_name_list_nudat = ["64Ganudat.csv","66Ganudat.csv","68Ganudat.csv","74Ganudat.csv","76Ganudat.csv","80Ganudat.csv","72Ganudat.csv","78Ganudat.csv"]
#file_name_measured = "AGa6000_1st_attempt.csv"
file_name_measured = "../AGa6000_4th_attempt.csv"

#file_name_measured = "APr6000_1st_attempt.csv"
#file_name_list_nudat = ["126Prnudat.csv","128Prnudat.csv","134Prnudat.csv","135Prnudat.csv","137Prnudat.csv","138Prnudat.csv","139Prnudat.csv","140Prnudat.csv","142Prnudat.csv","143Prnudat.csv", "145Prnudat.csv","146Prnudat.csv","147Prnudat.csv","150Prnudat.csv","152Prnudat.csv"]


# This part turns error values to the correct value of itself.
measured_centroids = measured_cent(file_name_measured)
katsayi = []
for number in measured_centroids:
    number = str(number).split(".")[1]
    katsayi.append(10**-len(number))
errorss = [7, 3, 7, 14, 10, 17, 6, 5, 4, 10, 3, 5, 3, 3,10, 5, 3, 6, 4, 14, 10, 9, 11, 14, 3, 4, 23, 12, 8, 7, 18, 14, 19, 24, 4,4,13,7]
errors = np.multiply(errorss,katsayi)
print("Errors: ",errors)
measured_data = measured_energy(file_name_measured)

print(measured_centroids)

############################

# Perform error propagation.

# Given values
a = slope  # Replace with your slope value from the linear fit
x = np.array(measured_centroids)
sigma_x = np.array(errors)

print("std of x: ",sigma_x)
sigma_a = slope_error  # Uncertainty in slope
sigma_b = intercept_error  # Uncertainty in intercept
print("uncertanity in slope: ",sigma_a,"\nuncertanity in intercept: ",sigma_b)
rho_ab = rho_ab  # Correlation coefficient between a and b
print("rho_ab: ",rho_ab)

# Calculate the uncertainty using the provided formula
sigma_f2 = (a**2) * (sigma_x**2) + (x**2) * (sigma_a**2) + sigma_b**2 + 2 * x * sigma_a * sigma_b * rho_ab
print(f"\nUncertainty (sigma_f): \n{np.sqrt(sigma_f2)}")

###########################
datam = {"Energy": np.array(measured_data),
         "Unc   ": np.array(np.sqrt(sigma_f2))
         }
df_unc = pd.DataFrame(datam)

print(df_unc)

print(f"Measured data: \n{measured_data}\n")


ALL_MATCHED_IN_MEASURED = []

for i in range(len(file_name_list_nudat)):
    MM = bulbul(file_name_list_nudat, file_name_measured)[i][1]
    ALL_MATCHED_IN_MEASURED.append(MM)

ALL_MATCHED_IN_MEASURED = np.sort(list(set(np.concatenate(ALL_MATCHED_IN_MEASURED))))

NOT_MATCHED_IN_MEASURED = [x for x in measured_energy(file_name_measured) if x not in ALL_MATCHED_IN_MEASURED]

print("ALL_MATCHED_IN_MEASURED: ",ALL_MATCHED_IN_MEASURED) # all measured which are in
                               # agreement with nudat data for all isotopes

print("\nNOT_MATCHED_IN_MEASURED:",NOT_MATCHED_IN_MEASURED) # data  did not matched with any of provided dataset.


BG_in_matched = np.sort(list(set(find_match(Background_all,ALL_MATCHED_IN_MEASURED,1.5)[0])))
print("\nBG_in_matched: ",BG_in_matched) # Data matched with background radiaton

BG_in_not_matched = np.sort(list(set(find_match(Background_all,NOT_MATCHED_IN_MEASURED,1.5)[0])))
print("\nBG_in_not_matched: ",BG_in_not_matched)

Unidentified_measured = [x for x in NOT_MATCHED_IN_MEASURED if x not in BG_in_not_matched]
print("\nUnidentified_measured: ",Unidentified_measured)# Shows unidentified data.


Write_info_element_wise(file_name_list_nudat,file_name_measured)
