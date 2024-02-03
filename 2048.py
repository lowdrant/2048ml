#!/usr/bin/env python3
from numpy import diff, fliplr, nonzero, uint32, zeros
from numpy.random import choice, randint


def _empty_left(row):
    """check if more left-shifting required:
        - any empty squares are to the left of numbered squares
        - any neighboring tiles that match
    """
    if all(row == 0):  # don't count empty rows
        return False
    if row[0] == 0:  # easy case: leftmost slot is empty
        return True
    ndx_fill = nonzero(row)[0]  # check if nonzero indices are contiguous
    if len(ndx_fill) > 1:
        if any(diff(ndx_fill) != 1):
            return True
    return False


def _slide_left(row):
    """slide all numbers left without combining"""
    if all(row == 0):
        return row
    while _empty_left(row):
        for j in range(len(row)-1):
            if row[j] == 0:
                row[j] = row[j+1]
                row[j+1] = 0
    return row


def _combine_left(row):
    """combine all numbers left; assumes no zeros inbetween"""
    sz = len(row)
    j = 0
    while j < sz-1:
        if row[j] == row[j+1]:
            row[j] += row[j+1]
            for k in range(j+1, sz-1):
                row[k] = row[k+1]
                row[k] = 0
            j += 1  # skip to next "pair"
        j += 1
    return row


def _move_left(grid):
    """apply left movement to 2048 grid"""
    for i in range(len(grid)):
        grid[i] = _slide_left(grid[i])
        grid[i] = _combine_left(grid[i])
    return grid


def _move_right(grid):
    return fliplr(_move_left(fliplr(grid)))


def _move_up(grid):
    return _move_left(grid.T).T


def _move_down(grid):
    return fliplr(_move_left(fliplr(grid.T))).T


class Grid:
    """Grid object for 2048 game. Grid is indexable and printable like a
        numpy array"""

    def __init__(self, sz=4):
        self.sz = sz
        self.reset()

    def reset(self):
        """reset grid to a game-start state"""
        self.grid = zeros((self.sz, self.sz), dtype=uint32)
        self._spawn()
        self._spawn()  # spawn 2 tiles to start

    def __format__(self, *args, **kwargs):
        return self.grid.__format__(*args, **kwargs)

    def __repr__(self, *args, **kwargs):
        return self.grid.__repr__(*args, **kwargs)

    def __str__(self, *args, **kwargs):
        return self.grid.__str__(*args, **kwargs)

    def __index__(self, *args, **kwargs):
        return self.grid.__index__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self.grid.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.grid.__setitem__(*args, **kwargs)

    def __call__(self, cmd, print_grid=False):
        # TODO: reject cmd if nothing to do in that dir
        self._apply_cmd(cmd)
        self._spawn()
        self._check_gameover()
        if print_grid:
            print(self)

    def _apply_cmd(self, cmd):
        """Update the grid by applying a direction movement command
            INPUT:
                cmd -- 0:left 1:right 2:up 3:down
        """
        if cmd == 0:
            transform = _move_left
        elif cmd == 1:
            transform = _move_right
        elif cmd == 2:
            transform = _move_up
        elif cmd == 3:
            transform = _move_down
        else:
            raise RuntimeError(f'Invalid cmd: {cmd}')
        self.grid = transform(self.grid)

    def _spawn(self):
        """spawn a new tile in an empty spot"""
        val = randint(0, 2)*2 + 2  # 2 or 4
        cands = nonzero(self.grid == 0)
        cand_ndx = choice(range(len(cands[0])))
        u, v = cands[0][cand_ndx], cands[1][cand_ndx]
        assert self.grid[u,
                         v] == 0, f'failed to spawn tile at nonempty {u},{v}'
        self.grid[u, v] = val

    def _check_gameover(self):
        """check if any neighboring tiles match, or if any tiles are empty"""
        col_same_mask = diff(self.grid, axis=0)
        row_same_mask = diff(self.grid, axis=1)
        if (col_same_mask == 0).any():
            return False
        if (row_same_mask == 0).any():
            return False
        if (self.grid == 0).any():
            return False
        return True


if __name__ == '__main__':
    g = Grid()
    print(g, '\n')
    for i in range(10):
        g(0, print_grid=1)
        print()