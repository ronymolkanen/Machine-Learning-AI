import pandas as pd  # Importing pandas for Excel operations
from tkinter import Frame, Label, CENTER
import logic
import constants as c
import AI_heuristics as h  # Importing the heuristic functions

class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.game_over = False
        self.start = True
        self.points = 0
        self.game_count = 0  # Counter for the number of games played
        self.scores = []  # List to store scores for each game

        self.grid()
        self.master.title('2048')
        
        self.commands = {
            c.KEY_UP: logic.up,
            c.KEY_DOWN: logic.down,
            c.KEY_LEFT: logic.left,
            c.KEY_RIGHT: logic.right,
            c.KEY_UP_ALT1: logic.up,
            c.KEY_DOWN_ALT1: logic.down,
            c.KEY_LEFT_ALT1: logic.left,
            c.KEY_RIGHT_ALT1: logic.right,
            c.KEY_UP_ALT2: logic.up,
            c.KEY_DOWN_ALT2: logic.down,
            c.KEY_LEFT_ALT2: logic.left,
            c.KEY_RIGHT_ALT2: logic.right,
        }

        self.grid_cells = []
        self.init_grid()
        self.reset_game()

        self.update_grid_cells()
        self.update_view()
        self.mainloop()

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2
                )
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def reset_game(self):
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = []
        self.transposition_table = {}
        self.points = 0
        self.game_over = False
        self.start = True

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number]
                    )
        self.update_idletasks()

    def dynamic_depth(self, max_tile):
        """Määrittää dynaamisen syvyyden suurimman laatan perusteella."""
        if max_tile < 131068:
            return 2
        else:
            return 4    

    def expectimax(self, matrix, depth, is_maximizing, alpha=-float('inf'), beta=float('inf')):
        if depth == 0 or logic.game_state(matrix) in ['win', 'lose']:
            return h.heuristic(matrix)

        if is_maximizing:
            best_score = -float('inf')
            for key in [c.KEY_UP, c.KEY_DOWN, c.KEY_LEFT, c.KEY_RIGHT]:
                new_matrix, done = self.commands[key](matrix)[:2]  # Simulate move
                if done:
                    score = self.expectimax(new_matrix, depth - 1, False, alpha, beta)
                    best_score = max(best_score, score)
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break  # Beta cut-off
            return best_score
        else:
            # Expectation step - simulate adding '2' or '4' to empty cells
            empty_cells = [(i, j) for i in range(c.GRID_LEN) for j in range(c.GRID_LEN) if matrix[i][j] == 0]
            if not empty_cells:
                return h.heuristic(matrix)

            expected_value = 0
            for i, j in empty_cells:
                new_matrix_2 = [row[:] for row in matrix]
                new_matrix_4 = [row[:] for row in matrix]
                new_matrix_2[i][j] = 2
                new_matrix_4[i][j] = 4

                score_2 = self.expectimax(new_matrix_2, depth - 1, True, alpha, beta)
                score_4 = self.expectimax(new_matrix_4, depth - 1, True, alpha, beta)

                expected_value += 0.9 * score_2 + 0.1 * score_4

            return expected_value / len(empty_cells)

    def AI_play(self):
        max_tile = max(max(row) for row in self.matrix)  # Get the maximum tile value
        depth = self.dynamic_depth(max_tile)  # Use the new dynamic depth method
        best_move = None
        best_score = -float('inf')
        for key in [c.KEY_UP, c.KEY_DOWN, c.KEY_LEFT, c.KEY_RIGHT]:
            new_matrix, done = self.commands[key](self.matrix)[:2]  # Simulate move
            if done:
                score = self.expectimax(new_matrix, depth, False)
                if score > best_score:
                    best_score = score
                    best_move = key
        return best_move

    def update_view(self):
        if not self.game_over:
            if self.start:
                self.start = False
                self.update_grid_cells()
                self.update()
            else:
                key = self.AI_play()
                if key:
                    self.matrix, done, points = self.commands[key](self.matrix)
                    self.points += points
                    if done:
                        self.matrix = logic.add_two(self.matrix)
                        self.history_matrixs.append(self.matrix)
                        self.update_grid_cells()
                        self.update()
                        if logic.game_state(self.matrix) == 'lose':
                            self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                            self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                            self.grid_cells[2][1].configure(text="Points:", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                            self.grid_cells[2][2].configure(text=self.points, bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                            self.game_over = True
                            print("You Lost! You got " + str(self.points) + " points")
                            self.scores.append(self.points)  # Append points to the scores list
                            self.game_count += 1
                            if self.game_count < 5:
                                self.after(1500, self.reset_and_start_new_game)  # Delay before starting a new game
                            else:
                                self.save_results_to_excel()  # Save results after 5 games
                                self.master.quit()  # Exit the application after 5 games

            self.update_grid_cells()
            self.update()
            self.after(5, self.update_view)  # Schedule the next update

    def reset_and_start_new_game(self):
        self.reset_game()
        self.update_view()

    def save_results_to_excel(self):
        """Tallentaa pelitulokset Exceliin ja laskee min, max sekä keskiarvo."""
        df = pd.DataFrame(self.scores, columns=["Score"])  # Create a DataFrame from scores
        df.loc["Min"] = df["Score"].min()
        df.loc["Max"] = df["Score"].max()
        df.loc["Average"] = df["Score"].mean()
        
        # Save the DataFrame to an Excel file
        df.to_excel("2048_game_results.xlsx", index_label="Game")
        print("Results saved to 2048_game_results.xlsx")

game_grid = GameGrid()