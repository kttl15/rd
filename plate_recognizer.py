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
    parser.add_argument("-i", "--image", help="Path to image.", required=False)
    # parser.add_argument(
    #     "-s",
    #     "--save",
    #     help="Save output if mode is retrive.",
    #     required=False,
    #     nargs="?",
    #     const=True,
    #     default=False,
    # )
    args = parser.parse_args()

    if not args.mode in ["i", "r"]:
        raise Exception("Invalid mode.")
    elif args.mode == "i" and not args.image:
        raise Exception("Infer selected but not image was given.")
    elif (
        args.mode == "i"
        and args.image
        and not os.path.isfile(os.path.join(os.getcwd(), args.image))
    ):
        raise Exception(f"File not found at {os.path.join(os.getcwd(), args.image)}")
    return args


def get_plate_pred(img):
    regions = ["nz"]
    API_TOKEN = json.load(open("plate_API.json", "r"))["API"]

    with open(img, "rb") as f:
        response = requests.post(
            "https://api.platerecognizer.com/v1/plate-reader/",
            data=dict(regions=regions),
            files=dict(upload=f),
            headers={"Authorization": f"Token {API_TOKEN}"},
        )
    return response


def save_pred_sqlite3(pred):
    connection = sqlite3.connect("pred.db")
    c = connection.cursor()
    c.execute(
        "INSERT INTO plate VALUES (?, ?)",
        [pred["results"][0]["plate"], json.dumps(pred)],
    )
    connection.commit()
    connection.close()
    print("Done.")


def retrive_pred():
    connection = sqlite3.connect("pred.db")
    c = connection.cursor()
    c = c.execute("SELECT * FROM plate")
    pprint(c.fetchall())
    connection.close()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "i":
        save_pred_sqlite3(get_plate_pred(args.image))
    elif args.mode == "r":
        retrive_pred()
