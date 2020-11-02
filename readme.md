# Scripts

1. car_jam.py -- does a request to carjam for each plate in the database and saves the results to the database.

2. create_db.py -- creates a sqlite database.

3. inference.py -- loads a Tensorflow model and infers the make and model of the image.

4. plate_recognizer.py -- does an api call for each image and save the results to an sqlite database.

5. train_model.py -- trains a Tensorflow model.

# Other

1. class_names.pickle -- file containing classes for the Tensorflow model.

---

## car_jam.py

1. calls a parameterized constructor with name of the database as the parameter.

2. calls **predict_from_db()** method.

3. **predict_from_db()** feteches data from the database using **\_\_get_data_from_db\_\_()** and returns a list of [img_names, plates, carjam].

4. for each img_name, plate, car_jam record, check if overwrite == true and car_jam record does not exist. If true, call **\_\_predict\_\_(name, plate)**

5. **\_\_predict\_\_(name, plate)** calls **\_\_parse_car_plate\_\_(plate)** which calls **\_\_get_data_car_jam\_\_(sleep_duration, plate)** and returns a PageElement. Then **\_\_parse_car_plate\_\_(plate)** returns a dict containing data of the car.

6. The dict is save to the database using the **\_\_save_car_jam_data\_\_()** method.

## inference.py

1. load images and do prefetching.

2. calls a parameterized constructor with path to model weights as the parameter.

3. calls **predict()** method on images.

4. gets the top 3 predictions of each image.

## plate_recognizer.py

1. use **argparse.ArgumentParser** to get input from cli.

2. calls a parameterized constructor with name of the database and arguments as the parameters.

3. calls **make_pred()** which calls **\_\_get_plate_pred\_\_()** which does an API call to plate recognizer. The response is parsed as json and saved to the database using the **\_\_save_pred_sqlite3()\_\_**.

## train_model.py

1. set gpu memory and logging level.

2. define path to images and load images. Do prefetching.

3. calls a parameterized constructor with img_shape as the parameter.

4. load base mode.

5. build new model.

6. compile model.

7. define necessary callbacks.

8. train model.
