import os
import datetime
import time
import requests
import re
import csv
import traceback

from dotenv import load_dotenv

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

script_dir = os.path.dirname(os.path.realpath(__file__))


def create_driver(default_download_dir):
    """
    Creates and returns a Selenium WebDriver instance with configured download settings.
    """
    # SETUP browser
    options = Options()
    options.BinaryLocation = "/snap/bin/chromium"
    options.add_experimental_option('prefs', {
        "download.default_directory": os.path.join(script_dir, default_download_dir),  # Change to your desired download folder
        "download.prompt_for_download": False,  # Disable download prompt
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })

    # tips for snap and chromium: https://stackoverflow.com/questions/5731953/use-selenium-with-chromium-browser
    # open chrome driver service
    chrome_service = Service(executable_path="/snap/bin/chromium.chromedriver")
    # open browser
    return webdriver.Chrome(service=chrome_service, options=options)


def login(driver):
    """
    Logs into the website using provided credentials.
    """
    # Load credentials from .env
    load_dotenv()
    login_email = os.getenv("DATAFLOW_EMAIL")
    login_pass = os.getenv("DATAFLOW_PASS")
    if not login_email or not login_pass:
        raise ValueError("Missing login credentials in the .env file.")

    # Fill in the username and password
    wait = WebDriverWait(driver, timeout=10)
    username_field = wait.until(EC.presence_of_element_located((By.ID, "email")))
    # username_field = driver.find_element(By.ID, "email")
    password_field = driver.find_element(By.ID, "password")
    login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")

    username_field.send_keys(login_email)
    password_field.send_keys(login_pass)
    time.sleep(0.5)
    login_button.click()
    print(f"Logged in.")


def fill_in_form(driver, date_interval, logger_group):
    """
    Fills in the date interval and selects the logger group from the dropdown.
    """
    # <select class="form-control m-b" name="report_loggers_dropdown" id="report_loggers_dropdown">
    #   <option>Select Logger</option><option value="4785">U11 - 364C - Multi Profile Soi Moisture and Temperature</option>
    #   ...
    # </select>
    # < select class ="form-control m-b group_selector" name="report_groups_dropdown" id="report_groups_dropdown" >
    #   < option > Select Group < / option >
    #   < option value="831" > Uhelná < / option >
    # ...
    #   < / select >
    wait = WebDriverWait(driver, timeout=10)
    location_dropdown = Select(wait.until(EC.presence_of_element_located((By.ID, "report_groups_dropdown"))))
    # there is a delay until all Select options are ready
    time.sleep(5)
    wait.until(EC.presence_of_element_located((By.XPATH, f"//option[text()='{logger_group}']")))
    # location_dropdown = Select(driver.find_element(By.ID, "report_groups_dropdown"))
    location_dropdown.select_by_visible_text(logger_group)
    time.sleep(0.5)

    # <input type="text" class="form-control hasDatepicker" id="report_from_date" placeholder="From" required="">
    # <input type="text" class="form-control hasDatepicker" id="report_to_date" placeholder="To" required="">
    date_from_input = driver.find_element(By.ID, "report_from_date")
    date_from_input.clear()
    dt = datetime.datetime.strptime(date_interval['start_date'], '%Y-%m-%d').date()
    dt_str = dt.strftime('%d-%m-%Y')
    date_from_input.send_keys(dt_str)

    date_to_input = driver.find_element(By.ID, "report_to_date")
    date_to_input.clear()
    dt = datetime.datetime.strptime(date_interval['end_date'], '%Y-%m-%d').date()
    dt_str = dt.strftime('%d-%m-%Y')
    date_to_input.send_keys(dt_str)


def gather_logger_info(driver, logger_group, download_dir):
    """
    Selects the logger group from the dropdown and goes through loggers.
    """
    # <select class="form-control m-b group_selector" name="groups_tab_groups_dropdown" id="groups_dropdown">
    #   <option>Select Group</option>
    #   <option selected="selected" value="601">Home</option>
    #   <option value="618">Lab</option>
    #   ...
    #   < / select >
    # <select class="form-control m-b" name="loggers_dropdown" id="loggers_dropdown">
    #   <option>Select Logger</option>
    #   <option value="3093">U01 - E9F7 - Multi Profile Soi Moisture and Temperature</option>
    #   ...
    #   </select>
    print(f"Gathering loggers info.")
    wait = WebDriverWait(driver, timeout=10)
    location_dropdown = Select(wait.until(EC.presence_of_element_located((By.ID, "groups_dropdown"))))
    # there is a delay until all Select options are ready
    time.sleep(5)
    wait.until(EC.presence_of_element_located((By.XPATH, f"//option[text()='{logger_group}']")))
    # location_dropdown = Select(driver.find_element(By.ID, "report_groups_dropdown"))
    location_dropdown.select_by_visible_text(logger_group)
    time.sleep(5)

    logger_dropdown = Select(driver.find_element(By.ID, "loggers_dropdown"))
    # time.sleep(5)
    # print(f"Waiting for logger_dropdown.")
    for i in range(60):
        if len(logger_dropdown.options)>1:
            # print(f"logger_dropdown ready.")
            break
        time.sleep(0.5)

    now = datetime.datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    # Iterate over each option in the logger dropdown
    print(f"Going through loggers.")
    rows = []
    for idx, option in enumerate(logger_dropdown.options):
        # Skip the "Select Logger" placeholder
        selected_text = option.text
        if selected_text == "Select Logger":
            continue

        print(f"{idx}/{len(logger_dropdown.options)-1} {selected_text[:10]}")

        logger_dropdown.select_by_visible_text(selected_text)
        time.sleep(0.5)

        # Extract number
        def extract_num(text):
            match = re.search(r"([-+]?\d*\.\d+|\d+)", text)
            if match:
                return match.group()
            else:
                return "0"

        # wait until the logger data are loaded
        # i.e. until the loggerUid is the same in dropdown and in info div
        # print(f"Waiting for logger_name.")
        for i in range(40):
            logger_name = driver.find_element(By.ID, "logger_name_info_box")
            # print(f"logger_name_info_box.")
            logger_type = driver.find_element(By.ID, "logger_type_info_box")
            # print(f"logger_type_info_box.")
            # resolve empty info div
            if ': ' not in logger_name.text or ': ' not in logger_type.text:
                time.sleep(0.5)
                continue
            text = logger_name.text.split(": ", 1)[1] + " - " + logger_type.text.split(": ", 1)[1]
            # print(f"compare text.")
            if selected_text == text:
                # print(f"logger_name ready.")
                break
            time.sleep(0.5)
            # print(f"next.")

        logger_uid = selected_text[:10].replace(" ", "")
        latitude_div = driver.find_element(By.ID, "logger_lat_info_box")
        longitude_div = driver.find_element(By.ID, "logger_lng_info_box")
        latitude = extract_num(latitude_div.text)
        longitude = extract_num(longitude_div.text)

        rows.append([current_date, current_time, logger_uid, latitude, longitude])

    with open(os.path.join(download_dir,f"logger_gps_list_{logger_group}.csv"), mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['date', 'time', 'loggerUid', 'latitude', 'longitude'])
        for r in rows:
            writer.writerow(r)



def submit_loggers(driver):
    """
    Iterates through loggers in the dropdown, submits reports, and confirms success.
    """
    logger_dropdown = Select(driver.find_element(By.ID, "report_loggers_dropdown"))
    time.sleep(1)
    logger_count = 0

    # Iterate over each option in the dropdown
    for option in logger_dropdown.options:
        # Skip the "Select Logger" placeholder
        if option.text == "Select Logger":
            continue

        logger_dropdown.select_by_visible_text(option.text)
        time.sleep(0.5)

        # <input class="btn btn-primary" id="btn_report_submit" type="submit" value="Submit">
        submit_btn = driver.find_element(By.ID, "btn_report_submit")
        submit_btn.click()
        # time.sleep(0.5)

        # wait for OK button
        # <div class="jconfirm-box></div>
        # <div class="jconfirm-title-c"><span class="jconfirm-icon-c"></span>
        # <span class="jconfirm-title">Processing Data... Please wait.</span></div>
        # <div class="jconfirm-content-pane" ...>
        # <div class="jconfirm-content" id="jconfirm-box34841">Press the refresh button shortly.</div></div>
        # <div class="jconfirm-buttons"><button type="button" class="btn btn-default">ok</button></div><div class="jconfirm-clear"></div></div>
        # Wait for the message box to appear
        wait = WebDriverWait(driver, timeout=10)
        message_box = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "jconfirm-box")))
        # Wait for the "OK" button inside the message box
        ok_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[text()='ok']")))
        ok_button.click()

        logger_count += 1
        # for testing:
        # if logger_count == 2:
        #     break

    return logger_count


def download_file(file_url, file_name, download_dir):
    """
    Handles downloading a file from a given URL and saves it to the specified directory.
    """
    try:
        print(f"Downloading {file_name} from {file_url}...")
        response = requests.get(file_url, timeout=30)
        if response.status_code == 200:
            file_path = os.path.join(download_dir, file_name)
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"Downloaded: {file_name}")
            return True
        else:
            print(f"Failed to download {file_name} (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {file_name}: {e}")
        return False


def download_reports(driver, download_dir, logger_count):
    """
    Iterates through table rows, retrieves file links, and downloads them.
    """
    # <table id="reports_table">
    #     <thead><tr role="row"><th class="sorting"...</th></tr></thead>
    #     <tbody><tr role="row" class="odd">
    #       <td><a href="https://clientreportsprocessed.s3.ap-southeast-2.amazonaws.com/jan.brezina@tul.cz:U11__364C:1738068349011.csv">
    #          jan.brezina@tul.cz:U11__364C:1738068349011.csv</a></td>
    #       <td class="sorting_1">2025-01-28 12:45:52 UTC</td><td>51824</td></tr><tr role="row" class="even">
    #     </tr></tbody></table>
    wait = WebDriverWait(driver, timeout=10)
    table = wait.until(EC.presence_of_element_located((By.ID, 'reports_table')))
    # table = driver.find_element(By.ID, "reports_table")

    # display 50 rows
    # <select name="reports_table_length" aria-controls="reports_table" class="">
    # <option value="10">10</option><option value="25">25</option><option value="50">50</option><option value="100">100</option>
    # </select>

    # Set display rows to 50
    num_entries_dropdown = Select(wait.until(EC.presence_of_element_located((By.NAME, 'reports_table_length'))))
    # num_entries_dropdown = Select(driver.find_element(By.NAME, "reports_table_length"))
    num_entries_dropdown.select_by_value("50")

    # Locate all rows inside the table's tbody
    rows = table.find_elements(By.XPATH, ".//tbody/tr")

    counter = 0
    for row in rows:
        link_element = row.find_element(By.XPATH, ".//td[1]/a")
        file_url = link_element.get_attribute("href")
        file_name = link_element.text.strip()

        res = download_file(file_url, file_name, download_dir)
        counter += 1

        if not res or counter == logger_count:
            break


def run_dataflow_extraction(download_dir, date_interval, logger_groups, flags):
    """
    Runs the full extraction process from login to downloading reports.
    """
    try:
        driver = create_driver(default_download_dir=download_dir)
        wait = WebDriverWait(driver, timeout=10)
        time.sleep(1)

        # Navigate to the website, returns once loaded
        driver.get('https://www.xpert.nz/home')

        login(driver)

        for logger_group in logger_groups:
            print(f"DataFlow extraction for group '{logger_group}'")

            if flags['location']:
                # wait for "Map" tab
                tab = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href="#tab-1"]')))
                tab.click()
                print(f"Tab '{tab.text}' Click")
                gather_logger_info(driver, logger_group, download_dir)
                # continue

            if flags['data_reports']:
                # wait for "Data reports" tab
                tab = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href="#reports"]')))
                # tab = driver.find_element(By.CSS_SELECTOR, 'a[href="#reports"]')
                print(f"Tab '{tab.text}' Click")
                tab.click()

                fill_in_form(driver, date_interval, logger_group)
                logger_count = submit_loggers(driver)
                # logger_count = 2  # testing without submitting
                # wait before refreshing the report links table
                time.sleep(15)

                # <input class="btn btn-success" id="btn_report_refresh" type="submit" value="Refresh">
                refresh_btn = driver.find_element(By.ID, "btn_report_refresh")
                refresh_btn.click()

                download_reports(driver, download_dir, logger_count)

    except Exception: traceback.print_exc()
    finally:
        print("DataFlow webpage extraction FINISHED")
        driver.quit()  # Ensure the browser is closed after the script runs


# Main script only defines parameters
if __name__ == '__main__':
    download_dir = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "_dataflow_grab"
    os.makedirs(download_dir)

    logger_groups = ["Lab", "Uhelná lesík", "Uhelná"]
    date_interval = {'start_date': '2025-01-01', 'end_date': '2025-04-22'}

    flags = {'location': True, 'data_reports': False}
    # Run the extraction
    run_dataflow_extraction(download_dir, date_interval, logger_groups, flags)
