from PIDSimulation import PIDSimulation
import argparse
import pandas as pd
# Initialize the argument parser
parser = argparse.ArgumentParser(description="Input a .csv file")

# Add an argument for the CSV file path
parser.add_argument('csv_file', type=str, help="Path to the .csv file")

# Parse the arguments
args = parser.parse_args()

# Read the CSV file using pandas
df = pd.read_csv(args.csv_file)

PIDSimulation([0.008, 0.0004, 0.06], df)
