import requests
import datetime
import csv
import io

# Set up your parameters and API information
url = "https://www.airnowapi.org/aq/observation/latLong/historical/"
latitude = 39.732
longitude = -105.015
distance = 15
api_key = "073EBBE6-6CC0-441B-ABC7-055A4ACE2302"


# Function to fetch data for a specific date
def fetch_data_for_date(date):
    params = {
        "format": "text/csv",
        "latitude": latitude,
        "longitude": longitude,
        "date": date,
        "distance": distance,
        "API_KEY": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Request for {date} failed with status code {response.status_code}")
        return None


# Define the start and end dates
start_date = datetime.date(2024, 1, 20)
end_date = datetime.date(2025, 1, 20)

# Path for the CSV file
csv_file_path = 'air_quality_data.csv'

# Open the CSV file in write mode to create it
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)

    # Loop through the date range
    current_date = start_date
    while current_date <= end_date:
        # Format the date into the required string format
        date_str = current_date.strftime('%Y-%m-%dT00-0000')

        # Fetch data for the current date
        data = fetch_data_for_date(date_str)

        if data:
            # Convert the raw CSV data into a file-like object
            csv_data = io.StringIO(data)
            reader = csv.reader(csv_data)

            # Skip the header of the data
            header_skipped = False
            for row in reader:
                if not header_skipped:
                    header_skipped = True
                    continue
                writer.writerow(row)
            print(f"Data for {date_str} added to file.")

        # Move to the next date
        current_date += datetime.timedelta(days=1)

print("Data collection complete.")
