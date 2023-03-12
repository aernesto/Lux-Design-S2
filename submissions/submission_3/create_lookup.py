from space import xy_iter
import json

if __name__ == '__main__':
    lookup = {}
    point2i = {}
    #1. loop over all points
    for i, self in enumerate(xy_iter(48)):
        point_dict = {'xys': (self.x, self.y, self.board_length)}
        # 2. write at_edge info
        point_dict['at_top_edge'] = self.y == 0
        point_dict['at_left_edge'] = self.x == 0
        point_dict['at_right_edge'] = self.x == self.board_length - 1
        point_dict['at_bottom_edge'] = self.y == self.board_length - 1
        #3. write all_neighbors
        point_dict['all_neighbors'] = [p.xy for p in self.all_neighbors]
        #4. write surrounding_neighbors
        point_dict['surrounding_neighbors'] = [p.xy for p in self.surrounding_neighbors]
        #5. write first lichen tiles
        point_dict['plant_first_lichen_tiles'] = [p.xy for p in self.plant_first_lichen_tiles]
        lookup[i] = point_dict
    with open('space_lookup.json', 'wt') as fh:
        json.dump(lookup, fh)
