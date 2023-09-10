
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from GameEnv import SoldierGameEnv

def visualize_game():
    game_env = SoldierGameEnv()
    fig, ax = plt.subplots()

    def update(num):
        ax.clear()
        game_env.step({"soldiers": []})  # Dummy actions, replace with actual game actions as needed

        # Get the information directly from the game environment object
        team_our = [(soldier.x, soldier.y) for soldier in game_env.teamOur.values()]
        team_enemy = [(soldier.x, soldier.y) for soldier in game_env.teamEnemy.values()]
        
        # Plot mountains (assuming mountains are represented by -1 in the game map)
        mountains = [(x, y) for x in range(game_env.width) for y in range(game_env.height) if game_env.map[x, y] == -1]
        if mountains:
            x, y = zip(*mountains)
            ax.scatter(x, y, c='grey', label='Mountain')
        
        # Plot our team
        x, y = zip(*team_our)
        ax.scatter(x, y, c='blue', label='Our Team')
        
        # Plot enemy team
        x, y = zip(*team_enemy)
        ax.scatter(x, y, c='red', label='Enemy Team')

        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=100, repeat=False)
    plt.show()

if __name__ == "__main__":
    visualize_game()
