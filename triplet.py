class Triplet:

    def __init__(self, doublet_1, doublet_2, triplet_id):
        """Class for Creating Triplets out of doublets from the Doublet class.
        :param doublet_1: object containing hit 1 and hit 2 of the now created triplet
        :param doublet_2: object containing hit 2 and hit 3 of the now created triplet
        :param triplet_id: unique integer id
        """
        self.doublet_1 = doublet_1
        self.doublet_2 = doublet_2
        self.triplet_id = triplet_id
        self.is_correct_match = self.set_is_correct_match()
        self.interactions = {}  # Interactions with other triplets
        self.quality = 0   # Quality in the sense of how good the triplet fits a chosen metric

    def angles_between_doublets(self):
        """Returns the angles between the doublets in xz and yz a a tuple
        :return
             (angle xz of doublets, angle yz of doublets)
        """
        angle_xz_doublet_1 = self.doublet_1.xz_angle()
        angle_yz_doublet_1 = self.doublet_1.yz_angle()

        angle_xz_doublet_2 = self.doublet_2.xz_angle()
        angle_yz_doublet_2 = self.doublet_2.yz_angle()

        return angle_xz_doublet_2 - angle_xz_doublet_1, angle_yz_doublet_2 - angle_yz_doublet_1

    def set_is_correct_match(self):
        """Checks if all hits of the triplets stem from the same particle.
        :return
            True if triplet originates from one single particle, else False.
        """
        if self.doublet_1.hit_1_particle_key == self.doublet_1.hit_2_particle_key == \
                self.doublet_2.hit_1_particle_key == self.doublet_2.hit_2_particle_key:
            return True
        return False