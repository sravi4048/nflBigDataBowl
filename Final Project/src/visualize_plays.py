import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import sys

def create_football_field(linenumbers=False,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12, 6.33)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
    return fig, ax

def plotPlayers(frame, passed_df):
    #col = passed_df['frameId']
    #max_index = col.idxmax()
    #print(frame)
    passed_df = passed_df[passed_df['frameId'] == frame]
    home = passed_df[passed_df['team'] == 'home']
    away = passed_df[passed_df['team'] == 'away']
    football = passed_df[passed_df['team'] == 'football']
    home_x = home['x']
    home_y = home['y']
    plt.scatter(home_x, home_y, s=30, c='Blue')
    away_x = away['x']
    away_y = away['y']
    plt.scatter(away_x, away_y, s=30, c='Red')
    football_x = football['x']
    football_y = football['y']
    plt.scatter(football_x, football_y, s=60, c='Brown')


def main(gameid, playid):
    if str(gameid)[:4] == '2018':
        tracking_df = pd.read_pickle("Data/2018trackingdata.pkl")
    elif str(gameid)[:4] == '2019':
        tracking_df = pd.read_pickle("Data/2019trackingdata.pkl")
    elif str(gameid)[:4] == '2020':
        tracking_df = pd.read_pickle("Data/2020trackingdata.pkl")
    else:
        print("wrong id")
        return
    game_tracking_data = tracking_df[(tracking_df['gameId'] == gameid) & (tracking_df['playId'] == playid)]
    #game_tracking_data = game_tracking_data['playid'] == playid
    col = game_tracking_data['frameId']
    max_index = col.idxmax()
    fig, ax = create_football_field()
    animator = ani.FuncAnimation(fig, plotPlayers, frames=range(1, max_index+1), fargs=[game_tracking_data], interval=100)
    plt.show()

main(int(sys.argv[1]), int(sys.argv[2]))
#main()