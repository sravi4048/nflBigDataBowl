# nflBigDataBowl

Data needs to be downloaded from site: https://www.kaggle.com/c/nfl-big-data-bowl-2022/data and stored in Data directory.

# src -- Primary files

# gunners.py
Contains main models. Running file from command line will produce information about the model as well as a list of top gunners.
More information about model can be accessed through object methods. Corresponding logistic regression models can also be accessed.

# top_long_snappers.py
Contains model for long snapper features, and produces list of top long snappers. Necessary data is pushed

# preprocess_data.py
Preprocesses data into form necessary for gunner models. When csvs are downloaded and pickled from Kaggle, they can be processed into necessary pickle files by calling this file.

# visualize_plays.py
Enables visualization of play if gameId and playId are passed as command line arguments.
E.g. call 'python3 src/visualize_plays.py 2018090600 37' to see play visualized.
Code for drawing football field is borrowed from: https://www.kaggle.com/robikscube/nfl-big-data-bowl-plotting-player-position

# src -- other files

# miscellaneous_exploration.py
Earlier models built--top gunners table currently built here.
