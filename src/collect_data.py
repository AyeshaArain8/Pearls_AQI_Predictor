import time
import subprocess
import sys


def collect_data():

    while True:

        print("Fetching new data...")

        # Run fetch_data.py using active virtual environment Python
        subprocess.run(
            [sys.executable, "src/fetch_data.py"]
        )


        print("Extracting features...")

        # Run feature_pipeline.py using active virtual environment Python
        subprocess.run(
            [sys.executable, "src/feature_pipeline.py"]
        )


        print("Data collection completed.")

        print("Waiting 2 minutes for next collection...\n")


        # Wait for 300 seconds
        time.sleep(120)



if __name__ == "__main__":

    collect_data()