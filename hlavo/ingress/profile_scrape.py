from pathlib import Path
import datetime
from hlavo.ingress.moist_profile.profile_scraper import run_dataflow_extraction

SCRIPT_DIR = Path(__file__).resolve().parent

# Main script only defines parameters
if __name__ == '__main__':

    download_dir = SCRIPT_DIR / "moist_profile" / \
        (datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "_dataflow_grab")
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    # logger_groups = ["Lab", "Uhelná lesík", "Uhelná"]
    # logger_groups = ["Uhelná lesík"]
    # logger_groups = ["Lab", "Uhelná lesík"]
    logger_groups = ["Uhelná"]
    # date_interval = {'start_date': '2025-08-01', 'end_date': '2025-08-31'}
    date_interval = {'start_date': '2025-11-01', 'end_date': '2025-11-30'}

    # flags = {'location': True, 'data_reports': False}
    flags = {'location': False, 'data_reports': True,
             'submit': True,
             # in case we do not submit, but only download
             # (grace_time: [min] how old are the table rows, we want to process)
             # 'submit': False, 'N_submissions': 24, 'grace_time': 55
             }
    # Run the extraction
    run_dataflow_extraction(download_dir, date_interval, logger_groups, flags)
