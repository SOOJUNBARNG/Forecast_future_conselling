import sys
import time
import schedule
import subprocess


def job():
    subprocess.run(["python", r"../data_download/clinic_rest_day_metabase.py"])
    time.sleep(10)
    subprocess.run(["python", r"../data_download/get_counsel_data.py.py"])
    time.sleep(10)
    subprocess.run(["python", r"arima_model_study_with_optuna.py"])
    time.sleep(10)
    subprocess.run(["python", r"arima_model.py"])
    time.sleep(10)
    subprocess.run(["python", r"sarima_model.py"])
    time.sleep(10)
    subprocess.run(["python", r"concate_model.py"])
    time.sleep(10)
    subprocess.run(["python", r"check_model.py"])
    time.sleep(10)


    return

if __name__ == "__main__":
    job()
