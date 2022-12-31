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

    def read_ratings(self):
        ratings = open(self.filename, 'r').readlines()
        data = np.array([[int(i) for i in rating[:-1].split("::")[:-1]] for rating in ratings])
        return data

    def generate_ui_matrix(self):
        data = np.zeros((self.num_users, self.num_movies))
        for rating in self.rating_tuples:
            data[rating[0] - 1][rating[1] - 1] = rating[2]
        return data

    def load_ui_matrix(self):
        if not os.path.exists(self.npy_file):
            ratings = self.generate_ui_matrix()
            np.save(self.npy_file, ratings)
        return np.load(self.npy_file)

    def __str__(self) -> str:
        return f"MovieLens1M DataLoader\nNumber of Users: {self.num_users}\nNumber of Movies: {self.num_movies}\nUI_Matrix: {self.ratings.shape}"


if __name__ == '__main__':
    dataloader = MovielensDatasetLoader()
    print(dataloader)
    print(dataloader.ratings)
