import datetime
import os

import numpy as np


class MovielensDatasetLoader:
    def __init__(self, filename='./ml-1m/ratings.dat', npy_file='./ml-1m/ratings.npy', num_movies=None, num_users=None):
        self.filename = filename
        self.npy_file = npy_file
        self.rating_tuples = self.read_ratings()
        if num_users is None:
            self.num_users = np.max(self.rating_tuples.T[0])
        else:
            self.num_users = num_users
        if num_movies is None:
            self.num_movies = np.max(self.rating_tuples.T[1])
        else:
            self.num_movies = num_movies
        self.ratings = self.load_ui_matrix()
        self.latest = self.generate_latest()

    def read_ratings(self):
        ratings = open(self.filename, 'r').readlines()
        data = np.array([[int(i) for i in rating[:-1].split("::")[:-1]] for rating in ratings])
        return data

    def generate_latest(self):
        latest_ratings = {}

        with open(self.filename, 'r') as f:
            for line in f:
                user_id, movie_id, rating, timestamp = map(int, line.strip().split("::"))
                timestamp = datetime.datetime.fromtimestamp(timestamp)
                user_id -= 1
                if user_id not in latest_ratings or timestamp > latest_ratings[user_id]["timestamp"]:
                    latest_ratings[user_id] = {"item_id": movie_id - 1,
                                               "timestamp": timestamp
                                               }
        return latest_ratings

    def generate_ui_matrix(self):
        data = np.zeros((self.num_users, self.num_movies))
        for rating in self.rating_tuples:
            data[rating[0] - 1][rating[1] - 1] = 1 if rating[2] > 0 else 0
        return data

    def load_ui_matrix(self):
        if not os.path.exists(self.npy_file):
            ratings = self.generate_ui_matrix()
            np.save(self.npy_file, ratings)
        return np.load(self.npy_file)


if __name__ == '__main__':
    dataloader = MovielensDatasetLoader()
    print(dataloader.ratings)
