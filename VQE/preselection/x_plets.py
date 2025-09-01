import csv
import time
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("../QCPatternRecognition/")

from preselection.doublet import Doublet
from preselection.triplet import Triplet
from preselection.segment import Segment
from preselection.segment_manager import SegmentManager


class XpletCreatorLUXE:
    def __init__(self,
                 luxe_tracking_data_file_name,
                 configuration,
                 save_to_folder):
        """
        Class for creating
        :param luxe_tracking_data_file_name: .csv file name with tracking data from the LUXE positron detector:
            hit_ID               : unique hit ID
            x                    : x value [m]
            y                    : y value [m]
            z                    : z value [m]
            layer_ID             : layer ID, starting from 0
            particle_ID          : number to identify particles
            particle_energy      : particle energy [MeV]
        :param configuration   : dictionary, configuration for detector setup and xplet selection
        :param save_to_folder  : folder in which results are stored
        """
        self.luxe_tracking_data_file_name = luxe_tracking_data_file_name
        self.configuration = configuration
        self.save_to_folder = save_to_folder

        # Manages segmentation, e.g. mapping dictionaries
        self.segment_manager = SegmentManager(configuration)

        # Simplified model only has 4 detector planes
        if len(configuration["detector planes"].keys()) == 4:
            self.simplified_model = True
        else:
            self.simplified_model = False
        self.num_layers = len(configuration["detector planes"].keys())

        # some values to check if computation successful
        self.num_particles = 0
        self.found_correct_doublets = 0
        self.found_correct_triplets = 0
        self.found_doublets = 0
        self.found_triplets = 0

        # indices from .csv file, code gets not messed up if some information might be added in the future to them
        self.x = None
        self.y = None
        self.z = None
        self.hit_id = None
        self.particle_id = None
        self.layer_id = None
        self.particle_energy = None

        # loading data and setting indices for .csv file
        self.num_segments = self.configuration["settings"]["number segments per layer"]
        if self.simplified_model:
            self.create_segments_simplified_model()
            self.segment_manager.map_segments_simple_model()
        else:
            pass  # to be implemented --> FullLUXE

        self.load_tracking_data()
        print(f"Number of particles found: {self.num_particles}")

        # storing triplets
        self.triplet_list = set()

        # Keeping track of coefficients and saving them for statistic purposes
        self.quality_wrong_match_list = []
        self.connectivity_wrong_match_list = []
        self.quality_correct_match_list = []
        self.connectivity_correct_match_list = []

        # Dictionary of quality and conflict functions
        self.quality_functions = {}
        self.conflict_functions = {}

        # Adding a very simple set of quality and conflict functions
        self.add_quality_function("two norm angle standard deviation", XpletCreatorLUXE.two_norm_std_angle)
        self.add_conflict_function("two norm angle standard deviation", XpletCreatorLUXE.two_norm_std_angle)

    def load_tracking_data(self):
        """
        Loads data from a .csv file and stores it into a 2-dim array. Also converts values which are used for
        calculations to float from string. The file structure is displayed in the class description.
        """
        particle_numbers = set()  # counting particle ID
        with open(self.luxe_tracking_data_file_name, 'r') as file:
            csv_reader = csv.reader(file)
            csv_header = next(csv_reader)  # access header, csv files should consist of one line of header
            self.x = csv_header.index("x")
            self.y = csv_header.index("y")
            self.z = csv_header.index("z")
            self.hit_id = csv_header.index("hit_ID")
            self.particle_id = csv_header.index("particle_ID")
            self.layer_id = csv_header.index("layer_ID")
            self.particle_energy = csv_header.index("particle_energy")
            for row in csv_reader:
                row_converted = []
                for i in range(len(row)):
                    if i in [self.x, self.y, self.z]:
                        row_converted.append(float(row[i]))  # convert coordinates from string to float
                    else:
                        row_converted.append(row[i])
                # storing in segment
                for segment in self.segment_manager.segment_list:
                    if float(row[self.z]) == segment.z_position:
                        if segment.x_start <= float(row[self.x]) <= segment.x_end:
                            segment.data.append(row_converted)
                particle_numbers.add(row[self.particle_id])
            self.num_particles = len(particle_numbers)

    def create_segments_simplified_model(self):
        """
        Splitting data into <splits> segments to reduce the combinatorial computational costs.
        For the simplified model only.
        """
        # Works just for the simplified model, the full setup has several chips per segment
        x_max = self.configuration["detector planes"]["layer 0"]["x end"]
        x_min = self.configuration["detector planes"]["layer 0"]["x start"]
        segment_size = (x_max - x_min) / self.num_segments

        for i, layer in enumerate(self.configuration["detector planes"].keys()):
            for j in range(self.num_segments):
                self.segment_manager.segment_list.append(Segment(f"L{i}S{j}",  # Layer i Segment j
                                                                 i,
                                                                 x_min + j * segment_size,
                                                                 x_min + (j + 1) * segment_size,
                                                                 self.configuration["detector planes"][layer]["z"]))

    def make_simplified_x_plet_list(self):
        """
        Creates and saves true doublet list and true triplet list. For the simplified model only.
        """
        print("\nCreating doublet lists...\n")
        doublet_list_start = time.time()  # doublet list timer
        for segment in self.segment_manager.segment_list:
            if segment.layer > 2:
                continue
            next_segments = self.segment_manager.target_segments(segment.name)
            for first_hit in segment.data:
                for target_segment in next_segments:
                    for second_hit in target_segment.data:
                        if self.doublet_criteria_check(first_hit[self.x],
                                                       second_hit[self.x],
                                                       first_hit[self.z],
                                                       second_hit[self.z]):
                            doublet = Doublet(first_hit[self.particle_id],
                                              (first_hit[self.x],
                                               first_hit[self.y],
                                               first_hit[self.z]),
                                              second_hit[self.particle_id],
                                              (second_hit[self.x],
                                               second_hit[self.y],
                                               second_hit[self.z]),
                                              first_hit[self.hit_id],
                                              second_hit[self.hit_id])
                            if doublet.is_correct_match:
                                self.found_correct_doublets += 1
                            self.found_doublets += 1
                            segment.doublet_data.append(doublet)

        doublet_list_end = time.time()  # doublet list timer
        print(f"Time elapsed for creating list of doublets: "
              f"{XpletCreatorLUXE.hms_string(doublet_list_end - doublet_list_start)}")
        print(f"Number of tracks approximately possible to reconstruct at doublet level: "
              f"{int(self.found_correct_doublets / 3)}")
        print(f"Number of doublets found: {self.found_doublets}\n")

        list_triplet_start = time.time()
        print("\nCreating triplet lists...\n")
        for segment in self.segment_manager.segment_list:
            if segment.layer > 1:
                continue
            next_segments = self.segment_manager.target_segments(segment.name)  # target segments
            for target_segment in next_segments:
                for first_doublet in segment.doublet_data:
                    for second_doublet in target_segment.doublet_data:
                        if first_doublet.hit_2_position != second_doublet.hit_1_position:  # check if match
                            continue
                        if self.triplet_criteria_check(first_doublet, second_doublet):
                            triplet = Triplet(first_doublet, second_doublet, self.found_triplets)
                            self.found_triplets += 1
                            segment.triplet_data.append(triplet)
                            if triplet.is_correct_match:
                                self.found_correct_triplets += 1

        list_triplet_end = time.time()

        print(f"Time elapsed for creating list of triplets: "
              f"{XpletCreatorLUXE.hms_string(list_triplet_end - list_triplet_start)}")
        print(f"Number of tracks approximately possible to reconstruct at triplet level: "
              f"{int(self.found_correct_triplets / 2)}")
        print(f"Number of triplets found: {self.found_triplets}")

    def triplet_criteria_check(self, doublet1, doublet2):
        """
        Checks if doublets may be combined to a triplet, depending on the doublet angles -> scattering
        :param doublet1: first doublet, nearer to IP
        :param doublet2: second doublet, further from IP
        :return: True if criteria applies, else False
        """
        if abs(doublet2.xz_angle() - doublet1.xz_angle()) < self.configuration["triplet criteria"]["angle diff x"]:
            if abs(doublet2.yz_angle() - doublet1.yz_angle()) < self.configuration["triplet criteria"]["angle diff y"]:
                return True
        return False

    def doublet_criteria_check(self, x1, x2, z1, z2):
        """
        Checks if hits may combined to doublets, applying dx/x0 criterion
        :param x1: x value first hit
        :param x2: x value second hit
        :param z1: z value first hit
        :param z2: z value second hit
        :return: True if criteria applies, else False
        """
        if abs(((x2 - x1) / self.z_at_x0(x1, x2, z1, z2) - self.configuration["doublet criteria"]["dx/x0 criteria"])) > \
                self.configuration["doublet criteria"]["dx/x0 criteria epsilon"]:
            return False
        return True

    def z_at_x0(self, x_end, x_start, z_end, z_start):
        """
        Help function for calculation x position of doublet at a z-reference value, usually the first detector layer
        counted from the IP
        :param x_end: x-position of target segment
        :param x_start: x-position of segment
        :param z_end: z-position of target segment
        :param z_start: z-position of segment
        :return: x_position at the reference layer
        """
        dx = x_end - x_start
        dz = z_end - z_start
        return x_end - dx * abs(z_end - self.configuration["doublet criteria"]["reference layer z"]) / dz

    def add_quality_function(self,
                             quality_function_name,
                             quality_function_object):
        """Adds a function to the quality function dictionary
        :param quality_function_name: name of the function
        :param quality_function_object:  function object
        """
        self.quality_functions.update({quality_function_name: quality_function_object})

    def add_conflict_function(self,
                              conflict_function_name,
                              conflict_function_object):
        """Adds a function to the conflict function dictionary
        :param conflict_function_name: name of the function
        :param conflict_function_object:  function object
        """
        self.conflict_functions.update({conflict_function_name: conflict_function_object})

    def set_triplet_coefficients(self):
        """
        Sets the triplet coefficients according to the configuration files. If a (re-)normalization was
        set it is also applied. If successful a message and the target folder location containing the triplet
        list is displayed.
        """
        for segment in self.segment_manager.segment_list:
            if segment.layer > 1:
                continue
            next_segments = self.segment_manager.target_segments(segment.name)  # target segments
            for t1 in segment.triplet_data:
                for target_segment in next_segments + [segment]:
                    if self.configuration["settings"]["constant quality"] is not None:
                        t1.quality = self.configuration["settings"]["constant quality"]

                    elif self.configuration["settings"]["quality from angles"]:
                        triplet_angles_xz, triplet_angles_yz = t1.angles_between_doublets()
                        t1.quality = np.sqrt(triplet_angles_xz**2 + triplet_angles_yz**2)

                # checking all combinations with other triplets
                    for t2 in target_segment.triplet_data:
                        interaction_value = self.triplet_interaction(t1, t2)
                        # Only interactions != 0 are treated
                        if interaction_value == 0:
                            continue
                        t1.interactions.update({t2.triplet_id: interaction_value})
                        t2.interactions.update({t1.triplet_id: interaction_value})

        # filling list structure
        for segment in self.segment_manager.segment_list:
            for triplet in segment.triplet_data:
                self.triplet_list.add(triplet)
            segment.triplet_data = []
        self.triplet_list = list(self.triplet_list)
        self.triplet_list.sort(key=lambda t: t.triplet_id)

        for t1 in self.triplet_list:
            # keeping track of statistics
            if t1.is_correct_match:
                self.quality_correct_match_list.append(t1.quality)
            else:
                self.quality_wrong_match_list.append(t1.quality)

            for i_key, i_value in t1.interactions.items():
                if i_key < t1.triplet_id:
                    t2 = self.triplet_list[i_key]
                    if t1.doublet_1.hit_1_particle_key == t1.doublet_1.hit_2_particle_key == \
                       t1.doublet_2.hit_1_particle_key == t1.doublet_2.hit_2_particle_key == \
                       t2.doublet_1.hit_1_particle_key == t2.doublet_1.hit_2_particle_key == \
                       t2.doublet_2.hit_1_particle_key == t2.doublet_2.hit_2_particle_key:
                        self.connectivity_correct_match_list.append(i_value)
                    else:
                        self.connectivity_wrong_match_list.append(i_value)

        # additional processing of qubo values
        if self.configuration["settings"]["z_scores"]:
            if self.configuration["settings"]["min_max feature scaling"] \
                    and self.configuration["settings"]["quality scale range"] is not None:
                print("Two standardisation methods not allowed at the same time!\n Aborting...")
                exit()
            else:
                quality_values = self.quality_correct_match_list + self.quality_wrong_match_list
                mu = np.mean(quality_values)
                sigma = np.std(quality_values)
                max_z_value = max([abs((value - mu) / sigma) for value in quality_values])
                for triplet in self.triplet_list:
                    triplet.quality = (triplet.quality - mu) / sigma / max_z_value

        # Scaling in the following way to [a, b] : X' = a + (X - X_min) (b - a) / (X_max - X_min)
        elif self.configuration["settings"]["min_max feature scaling"]:
            # scaling quality
            if self.configuration["settings"]["quality scale range"] is not None:
                a = self.configuration["settings"]["quality scale range"][0]
                b = self.configuration["settings"]["quality scale range"][1]
                quality_values = self.quality_correct_match_list + self.quality_wrong_match_list
                min_quality = min(quality_values)
                max_quality = max(quality_values)
                for triplet in self.triplet_list:
                    triplet.quality = a + (triplet.quality - min_quality) * (b - a) / (max_quality - min_quality)

                # rewriting ai lists
                for i in range(len(self.quality_correct_match_list)):
                    self.quality_correct_match_list[i] = \
                        a + (self.quality_correct_match_list[i] - min_quality) * (b - a) / (max_quality - min_quality)
                for i in range(len(self.quality_wrong_match_list)):
                    self.quality_wrong_match_list[i] = \
                        a + (self.quality_wrong_match_list[i] - min_quality) * (b - a) / (max_quality - min_quality)

            # scaling connectivity
            if self.configuration["settings"]["connectivity scale range"] is not None:
                # excluding conflict terms
                connectivity_values = [value for value in self.connectivity_correct_match_list if value !=
                                       self.configuration["settings"]["constant conflict param"]] + \
                                      [value for value in self.connectivity_wrong_match_list if value !=
                                       self.configuration["settings"]["constant conflict param"]]
                min_connectivity = min(connectivity_values)
                max_connectivity = max(connectivity_values)

                a = self.configuration["settings"]["connectivity scale range"][0]
                b = self.configuration["settings"]["connectivity scale range"][1]
                for triplet in self.triplet_list:
                    for key in triplet.interactions.keys():
                        if triplet.interactions[key] == self.configuration["settings"]["constant conflict param"]:
                            continue
                        triplet.interactions[key] = \
                            a + (triplet.interactions[key] - min_connectivity) * (b - a) / (
                                        max_connectivity - min_connectivity)

                # rewriting connectivity lists
                for i in range(len(self.connectivity_wrong_match_list)):
                    if self.connectivity_wrong_match_list[i] == \
                            self.configuration["settings"]["constant conflict param"]:
                        continue
                    self.connectivity_wrong_match_list[i] = \
                        a + (self.connectivity_wrong_match_list[i] - min_connectivity) * (b - a) / \
                        (max_connectivity - min_connectivity)
                for i in range(len(self.connectivity_correct_match_list)):
                    if self.connectivity_correct_match_list[i] == \
                            self.configuration["settings"]["constant conflict param"]:
                        continue
                    self.connectivity_correct_match_list[i] = \
                        a + (self.connectivity_correct_match_list[i] - min_connectivity) * (b - a) / \
                        (max_connectivity - min_connectivity)

        print("\nCoefficients set successfully")
        print(f"\nSaving triplet list to folder: {self.save_to_folder}")
        np.save(f"{self.save_to_folder}/triplet_list", self.triplet_list)

    def triplet_interaction(self,
                            triplet,
                            other_triplet):
        """
        Compares two triplets and  how they match.
        :param triplet: first triplet
        :param other_triplet: triplet to compare with the first one
        :return value based on connectivity/conflict and chosen set of parameters
        """

        # checking number of shared hits
        intersection = 0

        t1 = [triplet.doublet_1.hit_1_position,
              triplet.doublet_1.hit_2_position,
              triplet.doublet_2.hit_2_position]
        t2 = [other_triplet.doublet_1.hit_1_position,
              other_triplet.doublet_1.hit_2_position,
              other_triplet.doublet_2.hit_2_position]
        for value_t1 in t1:
            for value_t2 in t2:
                if value_t1 == value_t2:
                    intersection += 1

        # check if triplet origin from same layer
        if t1[0][2] == t2[0][2]:
            same_layer = True
        else:
            same_layer = False

        # same and not interacting triplets get a zero as a coefficient
        if intersection == 0 or intersection == 3:
            return 0

        # different mode types
        if self.configuration["settings"]["qubo mode"] == "c_con c_ct":
            if intersection == 1:
                return self.configuration["settings"]["constant conflict param"]
            elif intersection == 2 and not same_layer:
                return self.configuration["settings"]["constant connectivity param"]
            else:
                return self.configuration["settings"]["constant conflict param"]

        # constant for conflict, quality for function
        elif self.configuration["settings"]["qubo mode"] == "c_con f_ct":
            if intersection == 1:
                return self.configuration["settings"]["constant conflict param"]
            elif intersection == 2 and not same_layer:
                return - 1 + self.quality_functions[self.configuration["settings"]["function connectivity"]](
                    triplet.doublet_1, triplet.doublet_2, other_triplet.doublet_2)
            else:
                return self.configuration["settings"]["constant conflict param"]

        # function for conflict, constant for quality
        elif self.configuration["settings"]["qubo mode"] == "f_con c_ct":
            if intersection == 1:
                return self.conflict_functions[self.configuration["settings"]["function conflict"]](
                    triplet.doublet_1, triplet.doublet_2, other_triplet.doublet_2)
            elif intersection == 2 and not same_layer:
                return self.configuration["settings"]["constant connectivity param"]
            else:
                return 1 + self.conflict_functions[self.configuration["settings"]["function connectivity"]](
                    triplet.doublet_1, triplet.doublet_2, other_triplet.doublet_2)

        # only functions for conflicts / quality
        elif self.configuration["settings"]["qubo mode"] == "f_con f_ct":
            if intersection == 1:
                return self.conflict_functions[self.configuration["settings"]["function conflict"]](
                    triplet.doublet_1, triplet.doublet_2, other_triplet.doublet_2)
            elif intersection == 2 and not same_layer:
                return - 1 + self.quality_functions[self.configuration["settings"]["function connectivity"]](
                    triplet.doublet_1, triplet.doublet_2, other_triplet.doublet_2)
            else:
                return 1 + self.conflict_functions[self.configuration["settings"]["function conflict"]](
                    triplet.doublet_1, triplet.doublet_2, other_triplet.doublet_2)

    @staticmethod
    def hms_string(sec_elapsed):
        """Nicely formatted time string."""
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60
        return "{}:{:>02}:{:>05.2f}".format(h, m, s)

    @staticmethod
    def two_norm_std_angle(doublet_1, doublet_2, doublet_3):
        """Returns 2-norm of angle difference in xz and yz.
        :param
            doublet_1 : doublet from hit 1 + 2
            doublet_2 : doublet from hit 2 + 3
            doublet_3 : doublet from hit 3 + 4
        :return
            2-norm of angle difference in xz and yz."""
        angle_xz_doublet_1 = doublet_1.xz_angle()
        angle_yz_doublet_1 = doublet_1.yz_angle()

        angle_xz_doublet_2 = doublet_2.xz_angle()
        angle_yz_doublet_2 = doublet_2.yz_angle()

        angle_xz_doublet_3 = doublet_3.xz_angle()
        angle_yz_doublet_3 = doublet_3.yz_angle()

        return np.sqrt(np.std([angle_xz_doublet_1, angle_xz_doublet_2, angle_xz_doublet_3]) ** 2 +
                       np.std([angle_yz_doublet_1, angle_yz_doublet_2, angle_yz_doublet_3]) ** 2)

    def write_info_file(self):
        """
        Writes information about the Preselection parameters and some statistics into 'Preselection.txt' which is
        stored inside the target folder.
        """
        with open(self.save_to_folder + "/preselection_info.txt", "w") as f:
            f.write("Preselection performed with the following set of parameters: \n")
            f.write("---\n")
            f.write("detector planes: \n")

            for plane in self.configuration["detector planes"].keys():
                f.write(f"\n{plane}:")
                for key, value in self.configuration["detector planes"][plane].items():
                    f.write(f"\n{key}: {value}")
                f.write("\n")

            f.write("\n\n")
            f.write("---\n")
            f.write("doublet criteria: \n")
            for key, value in self.configuration["doublet criteria"].items():
                f.write(f"\n{key}: {value}")

            f.write("\n\n")
            f.write("---\n")
            f.write("triplet criteria: \n")
            for key, value in self.configuration["triplet criteria"].items():
                f.write(f"\n{key}: {value}")

            f.write("\n\n")
            f.write("---\n")
            f.write("settings: \n")
            for key, value in self.configuration["settings"].items():
                f.write(f"\n{key}: {value}")

            f.write("\n\n---\n")
            f.write("Statistics:\n\n")
            f.write(f"Number of particles hitting at leas one detector layer: {self.num_particles}\n\n")
            f.write(f"Number of doublets found: {self.found_doublets}\n")
            f.write(f"Number of tracks approximately possible to reconstruct at doublet level: "
                    f"{int(self.found_correct_doublets / 3)}\n\n")
            f.write(f"Number of triplets found: {self.found_triplets}\n")
            f.write(f"Number of tracks approximately possible to reconstruct at triplet level: "
                    f"{int(self.found_correct_triplets / 2)}\n")

    def plot_and_save_statistics(self):
        """
        This functions plots and saves various statistics to the same folder where the triplet list is saved to.
        The results are saved as 'triplet_coefficients_statistics.pdf' and 'triplet_interactions.pdf' into the
        target folder.
        """
        # Number of interactions with other triplets
        interactions_list = []
        for t in self.triplet_list:
            interactions_list.append(len(list(t.interactions.keys())))
        n, bins, _ = plt.hist(interactions_list, bins=max(interactions_list))
        plt.figure(figsize=(12, 9))
        plt.hist(np.array(interactions_list),
                 weights=1 / sum(n) * np.ones(len(interactions_list)),
                 bins=max(interactions_list),
                 edgecolor="firebrick",
                 linewidth=3,
                 histtype='step',
                 label=f"Number of particles: {self.num_particles}\n"
                       f"Number of triplets: {len(self.triplet_list)}")
        plt.yscale("log")
        plt.legend(loc="best", fontsize=20)

        plt.xlabel("Number of interactions with other triplets", fontsize=20)
        plt.ylabel("Fraction of counts", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(f"{self.save_to_folder}/triplet_interactions.pdf")

        # distribution of coefficients
        fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 12))

        ax1.hist([self.quality_correct_match_list, self.quality_wrong_match_list],
                 bins=50,
                 label=[r"quality correct match", r"quality wrong match"],
                 edgecolor='k',
                 color=["goldenrod", "royalblue"],
                 align="left")
        ax1.set_yscale('log')
        ax1.legend(loc='best', fontsize=20)
        ax1.tick_params(axis='both', labelsize=20)
        ax1.set_ylabel("counts", fontsize=20)
        ax1.set_xlabel("[a.u]", fontsize=20)

        n2, bins2, patches2 = ax2.hist([self.connectivity_correct_match_list, self.connectivity_wrong_match_list],
                                       bins=50,
                                       label=[r"connectivity correct match",
                                              r"connectivity wrong match"],
                                       edgecolor='k',
                                       color=["goldenrod", "royalblue"],
                                       align="left",
                                       rwidth=1)

        width = patches2[1][-1].get_width()
        height = patches2[1][-1].get_height()

        patches2[1][-1].remove()
        ax2.bar(1, align="center", height=height, width=width, color='red', edgecolor="k", label="conflicts")

        ax2.set_yscale('log')
        ax2.legend(loc='best', fontsize=20)
        ax2.set_ylabel("counts", fontsize=20)
        ax2.tick_params(axis='both', labelsize=20)
        ax2.set_xlabel("[a.u]", fontsize=20)
        plt.savefig(f"{self.save_to_folder}/triplet_coefficients_statistics.pdf")