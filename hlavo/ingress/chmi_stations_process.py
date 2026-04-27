from hlavo.ingress.meteo_playground.chmi_stations.data_processing import main as data_process

if __name__ == '__main__':
    start_date = "2020-01-01T00:00:00"
    end_date = "2025-12-31T23:59:59"

    data_process(start_date, end_date)
