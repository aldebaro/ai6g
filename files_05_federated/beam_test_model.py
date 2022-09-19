import tensorflow as tf
import numpy as np

from dataloader import LidarDataset2D
from models import Lidar2D

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--lidar_test_data", nargs='+', type=str, help="LIDAR test data file, if you want to merge multiple"
                                                                   " datasets, simply provide a list of paths, as follows:"
                                                                   " --lidar_training_data path_a.npz path_b.npz")
parser.add_argument("--beam_test_data", nargs='+', type=str, default=None,
                    help="Beam test data file, if you want to merge multiple"
                         " datasets, simply provide a list of paths, as follows:"
                         " --beam_training_data path_a.npz path_b.npz")
parser.add_argument("--model_path", type=str, help="Path, where the model is saved")
parser.add_argument("--preds_csv_path", type=str, default="unnamed_preds.csv",
                    help="Path, where the .csv file with the predictions will be saved")

args = parser.parse_args()


if __name__ == '__main__':
    test_data = LidarDataset2D(args.lidar_test_data, args.beam_test_data)

    test_data.lidar_data = np.transpose(test_data.lidar_data, (0, 2, 3, 1))

    model = Lidar2D
    model.load_weights(args.model_path)

    # metrics
    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_categorical_accuracy', dtype=None)
    top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_categorical_accuracy', dtype=None)
    top10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10_categorical_accuracy', dtype=None)

    model.compile(metrics=[top1, top5, top10])
    model.evaluate(test_data.lidar_data, test_data.beam_output)

    test_preds = model.predict(test_data.lidar_data, batch_size=100)

    np.savetxt(args.preds_csv_path, test_preds, fmt='%.5f', delimiter=', ')