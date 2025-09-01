import numpy as np


class Doublet:
    def __init__(self,
                 hit_1_particle_key,
                 hit_1_position,
                 hit_2_particle_key,
                 hit_2_position,
                 hit_1_id,
                 hit_2_id):
        """Class for doublet objects, consisting of two hits on different detector layers.
        :param hit_1_particle_key: particle true number, from simulation
        :param hit_1_position:  particle position [x, y, z] on detector layer of hit 1
        :param hit_2_particle_key: particle true number, from simulation
        :param hit_2_position: particle position [x, y, z] on detector layer of hit 2
        :param hit_1_id: unique integer hit id on detector of hit 1
        :param hit_2_id: unique integer hit id on detector of hit 2
        """
        self.hit_1_position = hit_1_position
        self.hit_2_position = hit_2_position
        self.hit_1_particle_key = hit_1_particle_key
        self.hit_2_particle_key = hit_2_particle_key
        self.hit_1_id = hit_1_id
        self.hit_2_id = hit_2_id
        self.is_correct_match = self.set_is_correct_match()

    def set_is_correct_match(self):
        """Checks if doublet hits stem from the same particle.
        :return
            True if created from same particle, else False.
        """
        if self.hit_1_particle_key == self.hit_2_particle_key:
            return True
        return False

    def xz_angle(self):
        """Returns the angle in the xz-plane with respect to beam axis in z direction.
        :return
            angle in the xz plane.
        """
        return np.arctan2((self.hit_2_position[0] - self.hit_1_position[0]),
                          (self.hit_2_position[2] - self.hit_1_position[2]))

    def yz_angle(self):
        """Returns the angle in the xz-plane with respect to beam axis in z direction.
        :return
            angle in the yz plane.
        """
        return np.arctan2((self.hit_2_position[1] - self.hit_1_position[1]),
                          (self.hit_2_position[2] - self.hit_1_position[2]))
