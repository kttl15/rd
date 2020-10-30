import requests
import json
import sqlite3
import time
from bs4 import BeautifulSoup


class CarJam:
    def __init__(self, db: str, overwrite: bool = False):
        self.connection = sqlite3.connect(db)
        self.c = self.connection.cursor()
        self.overwrite = overwrite

    def predict_from_db(self):
        """main method
        """
        img_names, plates, carjam = self.__get_data_from_db__()
        for name, plate, c in zip(img_names, plates, carjam):
            if self.overwrite:
                self.__predict__(name, plate)
            elif not self.overwrite and not c:
                self.__predict__(name, plate)
            else:
                print(f"Skipping for {name}")

        self.connection.commit()
        self.connection.close()

    def __predict__(self, name: str, plate) -> None:
        """get prediction from car jam

        Args:
        
            name (str): name of the image
            
            plate : output from plate recognizer
        """

        pred = json.loads(plate)
        if len(pred) and len(pred["results"]):
            plate = pred["results"][0]["plate"]
            print(f"Getting info for {name}.")
            self.__save_car_jam_data__(name, self.__parse_car_plate__(plate))
        else:
            self.__save_car_jam_data__(name, None)

    def __get_data_from_db__(self) -> list:
        """retrive data from database

        Returns:
        
            list: contains a list of image names (PK) plate recognizer outputs and carjam outputs
        """
        plate_recognizer_plate = self.c.execute("SELECT plate_recognizer FROM plate")
        plate_recognizer_plate = plate_recognizer_plate.fetchall()
        plate_recognizer_plate = [p[0] for p in plate_recognizer_plate]

        img_name = self.c.execute("SELECT id FROM plate")
        img_name = img_name.fetchall()
        img_name = [n[0] for n in img_name]

        carjam = self.c.execute("SELECT car_jam FROM plate")
        carjam = carjam.fetchall()
        carjam = [i[0] for i in carjam]

        return [img_name, plate_recognizer_plate, carjam]

    def __get_data_car_jam__(self, sleep_duration: int, plate: str):
        """Does a request to car jam.

        Args:
        
            sleep duration (int): sleep duration in seconds
            
            plate (str): plate number of the image

        Raises:
        
            Exception: request code error

        Returns:
        
            PageElement: table containing information of the car
        """
        results = requests.get(f"https://www.carjam.co.nz/car/?plate={plate}")
        if results.status_code == 200:
            print("Loading...")
            time.sleep(sleep_duration)
            soup = BeautifulSoup(results.content, "lxml")
            table = soup.find("table")
            return table
        else:
            raise Exception(f"Error with code {results.status_code}")

    def __parse_car_plate__(self, plate: str) -> dict:
        """parse data from car jam request

        Args:
        
            plate (str): plate number of the image

        Returns:
        
            dict: object containing data of the car
        """
        durations = [5, 10, 15, 20]
        count = 0
        while True:
            table = self.__get_data_car_jam__(durations[count], plate)
            if table:
                break
            else:
                if count < 3:
                    count += 1
        rows = table.find_all("tr")
        data = {}

        for row in rows:
            td = row.find("td")
            spans = td.find_all("span")
            key = spans[0].attrs["data-key"]
            value = spans[1].text
            data.update({key: value})

        return data

    def __save_car_jam_data__(self, img_name: str, data: dict) -> None:
        """save output of car jam to sqlite database

        Args:
        
            img_name (str): name of the image
            
            data (dict): output of car jam
        """
        if data:
            self.c.execute(
                f"UPDATE plate SET car_jam=? WHERE id=?", [json.dumps(data), img_name],
            )
            print("Done.")


jam = CarJam("pred.db")
jam.predict_from_db()

