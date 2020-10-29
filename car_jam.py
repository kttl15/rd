import requests
import json
import sqlite3
import time
from bs4 import BeautifulSoup


def parse_car_plate(plate: str):
    results = requests.get(f"https://www.carjam.co.nz/car/?plate={plate}")
    if results.status_code == 200:
        print("Loading...")
        time.sleep(10)
        soup = BeautifulSoup(results.content, "lxml")
        table = soup.find("table")
        rows = table.find_all("tr")
        data = {}

        for row in rows:
            td = row.find("td")
            spans = td.find_all("span")
            key = spans[0].attrs["data-key"]
            value = spans[1].text
            data.update({key: value})

        return data
    else:
        raise Exception(f"Error with code {results.status_code}")


def get_plate_from_db():
    connection = sqlite3.connect("pred.db")
    c = connection.cursor()
    plate_recognizer_plate = c.execute("SELECT plate_recognizer FROM plate")
    plate_recognizer_plate = c.fetchall()
    img_name = c.execute("SELECT id FROM plate")
    img_name = c.fetchall()
    connection.close()

    plates = []

    for name, p in zip(img_name, plate_recognizer_plate):
        pred = json.loads(p[0])
        name = name[0]
        if len(pred) and len(pred["results"]):
            plate = pred["results"][0]["plate"]
            print(f"Getting info for {name}.")
            save_car_jam_data(name, parse_car_plate(plate))
        else:
            save_car_jam_data(name, None)


def save_car_jam_data(img_name: str, data: dict):
    if data:
        connection = sqlite3.connect("pred.db")
        c = connection.cursor()
        c.execute(
            f"UPDATE plate SET car_jam=? WHERE id=?", [json.dumps(data), img_name],
        )
        connection.commit()
        connection.close()
        print("Done.")


# plate = "jsg421"
# data = parse_car_plate(plate)
# save_car_jam_data(plate, data)
get_plate_from_db()

