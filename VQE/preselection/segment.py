class Segment:
    def __init__(self, name, layer, x_start, x_end, z_position):
        """
        Handles segments of the detector layer
        :param name: string created from integer
        :param layer: 0-3 for simplified model, 0-7 for full setup
        :param x_start: start value of x of the segment [m]
        :param x_end: end value of x of the segment in x [m]
        :param z_position: position in z [m]
        """
        self.name = name
        self.layer = layer
        self.x_start = x_start
        self.x_end = x_end
        self.z_position = z_position
        self.data = []   # stored info from .csv file
        self.doublet_data = []   # doublet list, the start of the doublet is considered to be inside the segment
        self.triplet_data = []   # triplet list, the start of the triplet is considered to be inside the segment

    def is_inside_segment(self, x):
        """
        Checks if position (x, y) is inside the segment
        :param x: position in x [m]
        :return: True if (x, y) in segment, else False
        """
        if self.x_start <= x <= self.x_end:
            return True
        return False

    def add_entry(self, entry):
        """
        Adds an entry to segment
        :param entry: (x, y) or doublet objects
        """
        self.data.append(entry)