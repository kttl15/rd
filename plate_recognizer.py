import requests
import json
import argparse
import sqlite3
import os
from pprint import pprint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read license plates from images and output results as JSON.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-m", "--mode", help="Infer image (i) or retrive data (r)", required=False
    )
    parser.add_argument("-i", "--img", help="Path to frames.", required=False)
    # parser.add_argument(
    #     "-s",
    #     "--save",
    #     help="Save output if mode is retrive.",
    #     required=False,
    #     nargs="?",
    #     const=True,
    #     default=False,
    # )
    parser.add_argument(
        "-api",
        "--api",
        help="path to API Token in json format with 'API' as key and token as value. {'API': 'API_TOKEN'}",
        required=False,
    )
    args = parser.parse_args()

    if not args.mode in ["i", "r"]:
        raise Exception("Invalid mode.")
    elif args.mode == "i" and not args.img:
        raise Exception("Infer selected but not image was given.")
    elif (
        args.mode == "i"
        and args.img
        and (
            not os.path.isdir(os.path.join(os.getcwd(), args.img))
            or not os.path.isfile(os.path.join(os.getcwd(), args.api))
        )
    ):
        raise Exception(f"File not found at {os.path.join(os.getcwd(), args.img)}")
    return args


def retrive_pred():
    connection = sqlite3.connect("pred.db")
    c = connection.cursor()
    c = c.execute("SELECT * FROM plate")
    pprint(c.fetchall())
    connection.close()


class Recognizer:
    def __init__(self, db, args):
        self.args = args
        self.connection = sqlite3.connect(db)
        self.c = self.connection.cursor()
        self.ids = self.c.execute("SELECT id FROM plate")
        self.ids = [i[0] for i in self.ids]

    def make_pred(self):
        if os.path.isdir(self.args.img):
            path = os.path.join(os.getcwd(), self.args.img, "frames")
            imgs = sorted(os.listdir(path))
            for img in imgs:
                if not img in self.ids:
                    print(f"Recognizing {img}.")
                    self.save_pred_sqlite3(
                        img, self.get_plate_pred(os.path.join(os.getcwd(), path, img))
                    )
                else:
                    print(f"{img} exists. Skipping.")
        self.connection.commit()
        self.connection.close()

    def get_plate_pred(self, img):
        regions = ["nz"]
        API_TOKEN = json.load(open(self.args.api, "r"))["API"]

        with open(img, "rb") as f:
            response = requests.post(
                "https://api.platerecognizer.com/v1/plate-reader/",
                data=dict(regions=regions),
                files=dict(upload=f),
                headers={"Authorization": f"Token {API_TOKEN}"},
            )
        return response.json()

    def save_pred_sqlite3(self, img_name, pred):
        if pred:
            self.c.execute(
                "INSERT INTO plate VALUES (?, ?, ?, ?)",
                [img_name, json.dumps(pred), None, None],
            )
            print(f"Inserted {img_name}.")


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "i" and args.api:
        recognizer = Recognizer("pred.db", args)
        recognizer.make_pred()
    elif args.mode == "r":
        retrive_pred()
