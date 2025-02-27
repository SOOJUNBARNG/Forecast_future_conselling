# Required library
import os
import time
from datetime import datetime
import shutil
import glob

# Selenium related library
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

ID = "ban.sujun@medical-frontier.jp"
PW = "Ghostkill91!"
BASIC_URL = "https://metabase.medical-frontier.net/"
# TARGET_URL = "https://metabase.medical-frontier.net/question/5475"
# FILE_PATTERN = r"~/Downloads/【分析】直近３か月訪問者_*.csv"
# DIRECTORY = r"D:/SP-CK-recommation/RPA_run/"
# OUTPUT_FILE = r"metabase_basic_data/metabase_tvs_sql.csv"


def run_metabase(TARGET_URL, FILE_PATTERN, DIRECTORY, OUTPUT_FILE):
    # Path to your ChromeDriver
    # driver_path = Path(__file__).parent / "../Json_chrome/chromedriver.exe"
    # service = Service(str(driver_path))
    # driver = webdriver.Chrome(service=service)

    # Define Chrome options (optional)
    # Specify the ChromeDriver path
    chrome_driver_path = r"C:\Program Files\Chrome_driver\chromedriver.exe"

    # Create a Service object and pass it to webdriver
    service = Service(executable_path=chrome_driver_path)
    driver = webdriver.Chrome(service=service)

    try:
        driver.get(f"{BASIC_URL}")  # Or a URL if it's online
        time.sleep(1)

        # Wait for the element to be present before interacting with it
        wait = WebDriverWait(driver, 10)

        # Locate the id input field inside the form
        input_id_field = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="2"]')))
        input_id_field.send_keys(f"{ID}")

        # Locate the password input field inside the form
        pw_input = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="3"]')))
        pw_input.send_keys(f"{PW}")

        time.sleep(3)

        button = wait.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    '//*[@id="root"]/div/div/main/div/div[2]/div/div[2]/div/form/button/div',
                )
            )
        )
        button.click()

        date_int = datetime.today().strftime("%d")
        date_int = int(date_int)

        month_int = datetime.today().strftime("%m")
        month_int = int(month_int)

        year_int = datetime.today().strftime("%Y")
        year_int = int(year_int)

        driver.get(f"{TARGET_URL}")  # Or a URL if it's online

        # Wait for the element to be present before interacting with it
        wait = WebDriverWait(driver, 10)

        # ボタンをクリックする
        time.sleep(15)
        element_to_click = wait.until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    '//*[@id="root"]/div/div/main/div/div/div[2]/main/div[3]/div/div[3]/button/div/div',
                )
            )
        )
        element_to_click.click()

        for tippy_id in ["14", "13", "15", "16"]:
            try:
                csv_button_xpath = f'//*[@id="tippy-{tippy_id}"]/div/div/div/div/div/button[1]/div'
                element_to_click = wait.until(EC.element_to_be_clickable((By.XPATH, csv_button_xpath)))
                element_to_click.click()
                time.sleep(50)  # Allow file download
                break  # Exit loop once the click is successful
            except Exception as e:
                print(f"Failed to click button with tippy-{tippy_id}: {e}")

        expanded_file_pattern = os.path.expanduser(f"{FILE_PATTERN}")

        # List all matching files
        csv_files = glob.glob(expanded_file_pattern)
        target_directory = os.path.expanduser(f"{DIRECTORY}")

        # Ensure target directory exists, create it if not
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        if csv_files:
            # Get the most recent file based on the modification time
            latest_file = max(csv_files, key=os.path.getmtime)
            # Define the full path of the new filename in the target directory
            new_filename = os.path.join(target_directory, f"{OUTPUT_FILE}")
            print(latest_file)
            print(new_filename)
            # Move and rename the file
            shutil.move(latest_file, new_filename)

        else:
            print("No files found matching the pattern.")

    finally:
        # Close the driver after the operation (optional, depending on your workflow)
        driver.quit()


def main(TARGET_URL, FILE_PATTERN, DIRECTORY, OUTPUT_FILE):
    run_metabase(TARGET_URL, FILE_PATTERN, DIRECTORY, OUTPUT_FILE)  # Calling the run_metabase function within main()


# if __name__ == "__main__":
#     main(TARGET_URL, FILE_PATTERN, DIRECTORY, OUTPUT_FILE)  # Ensures main is called when the script is executed
