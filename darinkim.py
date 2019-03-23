import logging
import os
import random

import backend
import numpy as np
import renju

LETTERS = 'abcdefghjklmnop'
NUMBERS = [i for i in range(1, 16)]

turn = 0


class Position:
    def __init__(self, turn, seq, label=None):
        self.fir = np.array([[0 for j in range(15)] for i in range(15)])
        self.sec = np.array([[0 for j in range(15)] for i in range(15)])
        self.thi = np.array([[turn for j in range(15)] for i in range(15)])
        self.label = label
        for i in range(len(seq)):
            if i % 2 == 0:
                self.fir[seq[i][0]][seq[i][1]] = 1
            else:
                self.sec[seq[i][0]][seq[i][1]] = -1
        self.image = np.array([self.fir, self.sec, self.thi])



def translate(coords):    # type(coords) == str: k8, h1, etc. res is a tuple
    coordFir = LETTERS.index(coords[0])
    coordSec = int(coords[1:]) - 1
    return (coordSec, coordFir)

def translate2(coords):   #type(coords) == tuple: (7, 7), (9, 9), etc. res is number of class
    res = coords[0] * 15 + coords[1]
    return res

def to_seq(game):
    seq = []
    for i in range(len(game)):
        seq.append(translate(game[i]))
    return seq


    
def choose_clever_move(board):
    positions = renju.list_positions(board, renju.Player.NONE)
    positions2 = positions[:-2] #<<<<<<<<<<<<< ask have I to destroy \n at the end
    positionsList = list(map(str, positions2.split()))
    if turn % 2 == 0:
        curPlayer = 1
    else:
        curPlayer = -1
    
    
    curTablePos = Position(turn=curPlayer, seq=to_seq(positionsList))
    
    win_condition = 1 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DONT FORGET
    
    table_for_model = np.empty((15, 15, 3))
    for i in range(3):
        table_for_model[:,:,i] = curTablePos.image[i,:,:]
    
    y = model.predict_proba(np.array([table_for_model]))
    
    answer_class = np.argmax(y)
    answer_tuple = (answer_class // 15, answer_class % 15)
    answer_coords = LETTERS[answer_tuple[1]] + str(NUMBERS[answer_tuple[0]])
    
    return answer_coords

def main():
    '''
    pid = os.getpid()
    LOG_FORMAT = str(pid) + ':%(levelname)s:%(asctime)s: %(message)s'

    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
    logging.debug("Start dummy backend...")
    '''
    model = load_model('night_model.hdf5')

    while True:
        '''
        logging.debug("Wait for game update...")
        game = backend.wait_for_game_update()

        if not game:
            logging.debug("Game is over!")
            return

        logging.debug('Game: [%s]', game.dumps())
        '''
        boardStr = sys.stdin.readline()
        if not boardStr:
            break

        move = choose_clever_move(boardStr)

        '''
        if not backend.set_move(move):
            logging.error("Impossible set move!")
            return

        logging.debug('Random move: %s', move)
        '''
        sys.stdout.write(move + '\n')
        sys.stdout.flush()


if __name__ == "__main__":
    main()
