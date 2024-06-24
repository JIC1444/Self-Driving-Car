import matplotlib.pyplot as plt
from matplotlib.pyplot import grid
import numpy as np

path = []

class Node():
    def __init__(self ,parent = None, position = None):
        self.parent = parent
        self.position = position
        self.f = 0
        self.g = 0
        self.h = 0

    def __eq__(self,other):
        return self.position == other


def a_star_search(grid, start, end):
    open_list = []
    closed_list = []

    start_node = Node(None, start)
    end_node = Node(None, end)
    open_list.append(start_node)

    while solved == False:
        current_node = open_list[0]
        
        if current_node == end_node:
            print("End node found!\nCalculating path:")
            solved == True
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            print(path[::-1])
            return path
        
        children = []

        #find children, ignores untraversable nodes and nodes outside of grid
        for new_pos in [(0, -1), (0, 1), (-1, 0),(1, 0), (-1, -1), (-1, 1),(1, -1), (1, 1)]:
            child_node_pos = [current_node.position[0] + new_pos[0],
                    current_node.position[1] + new_pos[1]]

            if child_node_pos[0] > (len(grid) - 1) or child_node_pos[0] < 0 or child_node_pos[1] > (len(grid[len(grid)-1]) -1) or child_node_pos[1] < 0:
                continue

            if grid[child_node_pos[0]][child_node_pos[1]] != 0:
                continue

            child_node = Node(current_node,child_node_pos)
            children.append(child_node)

        open_list_temp = []
        print(children,"children")
        lowest_f = 999999999999

        for search in children:
            search.g = current_node.g
            horiziontal = end[0] - search.position[0]
            vertical = end[1] - search.position[1]
            search.h = horiziontal**2 + vertical**2
            search.g = search.g + 1  
            search.f = search.h + search.g 
            print(search.f,"search f") 
            open_list_temp.append(search)
            if search.f < lowest_f:
                lowest_f = search.f
                for y in range(1,len(open_list)-2):
                    open_list.remove(open_list[y])
                open_list.append(search)
        
        closed_list.append(open_list[0]), open_list.remove(open_list[0])
        print("open_list", open_list)


def main():
    grid = [[0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0]
            ]
    
    start =[0,0]
    end = [7,8]
    path = a_star_search(grid,start,end)

    #visualising the path

    image = np.array(grid)
    for idx in path:
        idx[0] = idx[0] - 1
        idx[1] = idx[1] - 1
        image[idx] = 1
        print(image)
    plt.matshow(image)
    plt.show()

solved = False
if __name__ == "__main__":
    main()

