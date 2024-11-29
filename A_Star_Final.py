import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Ryhmä GigaChads / Rony Mölkänen, Samuli Ronni, Mikael Rokkanen, Samppa Saarinen

class Node:
    """Solmu-luokka A*-reittihakuun"""

    def __init__(self, parent=None, position=None):
        # Solmun vanhempi (edellinen solmu reitillä)
        self.parent = parent
        # Solmun sijainti (koordinaatit)
        self.position = position
        # G-arvo: etäisyys lähtösolmusta nykyiseen solmuun
        self.g = 0
        # H-arvo: heuristinen arvio etäisyydestä kohdesolmuun
        self.h = 0
        # F-arvo: G- ja H-arvojen summa, joka määrittää solmun "kokonaiskustannuksen"
        self.f = 0

    def __eq__(self, other):
        # Vertaa solmujen sijaintia, palauttaa True, jos sijainnit ovat samat
        return self.position == other.position

    def __repr__(self):
        # Palauttaa solmun merkkijonoesityksen
        return f"Node(position={self.position}, f={self.f}, g={self.g}, h={self.h})"

# Suunnat ja niiden koordinaattimuutokset
UP, RIGHT, DOWN, LEFT = (-1, 0), (0, 1), (1, 0), (0, -1)
DIRECTION_NAMES = {UP: "YLÖS", RIGHT: "OIKEA", DOWN: "ALAS", LEFT: "VASEN"}

# Sallitut liikkeet eri suuntiin (virhekäännösten estämiseksi)
Allowed_Moves_for_Direction = {
    UP: [UP, RIGHT],
    RIGHT: [RIGHT, DOWN],
    DOWN: [DOWN, LEFT],
    LEFT: [LEFT, UP]
}

def heuristic(a, b):
    """Heuristiikkafunktio: laskee euklidisen etäisyyden neliön"""
    return (a[0] - b[0])**2 + (a[1] - b[1])**2 
    
def astar(maze, start, end):
    """Palauttaa listan tuplista poluksi start- ja end-pisteen välillä annetussa labyrintissä"""

    # Luodaan aloitus- ja lopetussolmut
    start_node = Node(None, start)
    end_node = Node(None, end)
    open_list = [start_node]  # Lista solmuista, joita pitää vielä käsitellä
    closed_set = set()  # Suljettu joukko jo käsitellyistä solmuista
    came_from = {}  # Reitti solmuista
    g_cost = {}  # G-arvot solmuille
    
    came_from[start_node.position] = None
    g_cost[start_node.position] = 0

    while open_list:
        # Valitaan solmu, jolla on pienin f-arvo
        current_node = min(open_list, key=lambda node: node.f)
        open_list.remove(current_node)
        closed_set.add(current_node.position)

        # Jos nykyinen solmu on kohdesolmu, palautetaan polku ja suuntien lista
        if current_node == end_node:
            path = []
            directions = []
            while current_node is not None:
                path.append(current_node.position)
                if current_node.parent is not None:
                    move = (current_node.position[0] - current_node.parent.position[0],
                            current_node.position[1] - current_node.parent.position[1])
                    direction_str = DIRECTION_NAMES.get(move, "TUNNISTAMATON")
                    directions.append(direction_str)
                current_node = current_node.parent
            return simplify_path(path[::-1]), directions[::-1]

        # Sallitut liikkeet nykyisestä suunnasta
        allowed_moves = [UP, RIGHT, DOWN, LEFT]
        if current_node.parent is not None:
            move = (current_node.position[0] - current_node.parent.position[0],
                    current_node.position[1] - current_node.parent.position[1])
            allowed_moves = Allowed_Moves_for_Direction.get(move, allowed_moves)

        # Käsitellään kaikki sallitut liikkeet
        for move in allowed_moves:
            node_position = (current_node.position[0] + move[0], current_node.position[1] + move[1])
            if (node_position[0] >= len(maze) or node_position[0] < 0 or
                node_position[1] >= len(maze[0]) or node_position[1] < 0):
                continue  # Jos ollaan labyrintin ulkopuolella, siirrytään seuraavaan liikkeeseen

            if maze[node_position[0]][node_position[1]] != 0:
                continue  # Jos kohde ei ole käveltävissä, siirrytään seuraavaan liikkeeseen

            new_node = Node(current_node, node_position)
            if new_node.position in closed_set and g_cost.get(new_node.position, float('inf')) <= new_node.g:
                continue  # Jos solmu on jo käsitelty ja uusi reitti on kalliimpi, siirrytään seuraavaan liikkeeseen

            # Päivitetään solmun G-, H- ja F-arvot
            new_node.g = current_node.g + 1
            new_node.h = heuristic(new_node.position, end_node.position)
            new_node.f = new_node.g + new_node.h

            # Tarkistetaan, onko uusi solmu jo avoimessa listassa paremmalla F-arvolla
            if any(new_node == open_node and new_node.f >= open_node.f for open_node in open_list):
                continue

            # Lisätään uusi solmu avoimeen listaan ja päivitetään reitti ja g-kustannus
            open_list.append(new_node)
            came_from[new_node.position] = current_node.position
            g_cost[new_node.position] = new_node.g

    return [], []  # Jos polkua ei löydy, palautetaan tyhjä lista

def simplify_path(path):
    """Poistaa turhat käännökset polusta"""
    if not path:
        return []

    new_path = [path[0]]
    for i in range(1, len(path) - 1):
        prev = new_path[-1]
        curr = path[i]
        next = path[i + 1]

        # Tarkista, onko suunta muuttunut
        if (curr[0] - prev[0], curr[1] - prev[1]) != (next[0] - curr[0], next[1] - curr[1]):
            new_path.append(curr)

    new_path.append(path[-1])
    return new_path

def visualize_maze_and_path(maze, path):
    """Visualisoi labyrintin ja reitin reaaliaikaisena animaationa matplotlibin avulla"""

    def add_arrow(ax, start, direction, color='black'):
        """Lisää nuoli kuvioon."""
        arrow = patches.FancyArrowPatch(
            (start[1], start[0]),  # Alkuperäinen solmu
            (start[1] + 0.5 * direction[1], start[0] + 0.5 * direction[0]),  # Nuolen pää
            mutation_scale=20,
            color=color,
            arrowstyle='->',
            linewidth=3
        )
        return arrow

    # Muunna labyrintti numpy-taulukoksi
    maze_array = np.array(maze)

    # Luo kuva ja akselit
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Näytä labyrintti
    im = ax.imshow(maze_array, cmap='gray_r', origin='upper')

    # Luo reitin animaatio
    path_x, path_y = zip(*path)
    path_x = list(path_x)
    path_y = list(path_y)

    # Luodaan tyhjät listat animaatiota varten
    path_line, = ax.plot([], [], 'o-', color='red', markersize=5, linestyle='-', linewidth=2)
    current_point, = ax.plot([], [], 'o', color='blue', markersize=10)
    arrows = []

    def init():
        path_line.set_data([], [])
        current_point.set_data([], [])
        for arrow in arrows:
            arrow.remove()
        arrows.clear()
        return path_line, current_point

    def update(frame):
        # Päivitä reitin osuus
        path_line.set_data(path_y[:frame], path_x[:frame])
        # Päivitä nykyinen kohta
        current_point.set_data(path_y[frame], path_x[frame])
        
        # Lisää nuolia käännöksiin
        for arrow in arrows:
            arrow.remove()
        arrows.clear()
        
        if frame > 1:
            prev = (path_x[frame - 2], path_y[frame - 2])
            current = (path_x[frame - 1], path_y[frame - 1])
            next = (path_x[frame], path_y[frame])
            
            direction = (next[0] - current[0], next[1] - current[1])
            prev_direction = (current[0] - prev[0], current[1] - prev[1])
            
            if direction != prev_direction:
                # Lisää nuoli käännökseen
                arrow = add_arrow(ax, current, direction, color='blue')
                arrows.append(arrow)
                ax.add_patch(arrow)
                
        return path_line, current_point

    ani = FuncAnimation(fig, update, frames=len(path_x), init_func=init, blit=True, interval=500)

    plt.grid(which='both', color='black', linestyle='-', linewidth=1)
    plt.title('Labyrintti ja reitin eteneminen')
    plt.show()

def main():
    # Labyrintti, jota etsitään ratkaisua (1 = seinä, 0 = käytävä)
    maze = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 1, 0 ,1, 1, 1, 0, 1, 1, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 1, 0 ,1, 1, 1, 1, 1, 0, 1, 0],
             [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
             [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
             [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
             [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
             [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]]

    start = (11, 10)  # Lähtöpiste
    end = (11, 2)  # Päätepiste

    # Käytetään A*-algoritmia reitin löytämiseksi
    path, directions = astar(maze, start, end)
    if path:
        print("Polku löytyi:", path)
        print("Suunnat:", directions)
        # Visualisoidaan labyrintti ja löydetty reitti
        visualize_maze_and_path(maze, path)
    else:
        print("Polkua ei löytynyt.")

if __name__ == "__main__":
    main()
