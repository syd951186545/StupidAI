
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from GameEnv import SoldierGameEnv

def visualize_game():
    game_env = SoldierGameEnv()
    fig, ax = plt.subplots()

    def update(num):
        ax.clear()
        game_state,_,_ = game_env.render()
        game_map = game_state["mapInfo"]["zones"]
        team_our = game_state["players"]["teamOur"]["roles"]
        team_enemy = game_state["players"]["teamEnemy"]["posList"]
        
        # Plot mountains
        mountains = [(zone['pos']['x'], zone['pos']['y']) for zone in game_map if zone['roleType'] == 'mountain']
        if mountains:
            x, y = zip(*mountains)
            ax.scatter(x, y, c='grey', label='Mountain')
        
        # Plot our team
        x, y = zip(*[(soldier['pos']['x'], soldier['pos']['y']) for soldier in team_our])
        ax.scatter(x, y, c='blue', label='Our Team')
        
        # Plot enemy team
        x, y = zip(*[(soldier['pos']['x'], soldier['pos']['y']) for soldier in team_enemy])
        ax.scatter(x, y, c='red', label='Enemy Team')

        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=100, repeat=False)
    plt.show()

if __name__ == "__main__":
    visualize_game()
