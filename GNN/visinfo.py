import pandas as pd


def build_board_game_dataset():
    ratings_url = (
        'https://raw.githubusercontent.com/rfordatascience/tidytuesday/'
        'main/data/2022/2022-01-25/ratings.csv'
    )
    details_url = (
        'https://raw.githubusercontent.com/rfordatascience/tidytuesday/'
        'main/data/2022/2022-01-25/details.csv'
    )

    ratings = pd.read_csv(ratings_url)
    details = pd.read_csv(details_url)
    games = pd.merge(ratings, details, on='id')

    selected_games = games[
        (games['yearpublished'] >= 2000)
        & (games['users_rated'] > 100)
        & (games['playingtime'] > 0)
        & (games['playingtime'] <= 240)
    ].copy()

    categories = selected_games['boardgamecategory'].astype(str)
    categories = categories.str.replace("[", "", regex=False)
    categories = categories.str.replace("]", "", regex=False)
    categories = categories.str.replace("'", "", regex=False)
    categories = categories.str.split(',').str[0].str.strip()
    selected_games['main_category'] = categories

    columns = [
        'name',
        'yearpublished',
        'average',
        'users_rated',
        'playingtime',
        'main_category'
    ]
    result = selected_games[columns]
    result.to_csv('board_games_lite.csv', index=False)


if __name__ == '__main__':
    build_board_game_dataset()
