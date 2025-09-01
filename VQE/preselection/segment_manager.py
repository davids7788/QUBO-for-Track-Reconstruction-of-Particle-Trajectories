import pandas as pd


class SegmentManager:
    def __init__(self, selection_criteria):
        """
        Class for handling segments for doublet and triplet creation
        :param selection_criteria:
        """
        self.selection_criteria = selection_criteria   # Dictionary of selection criteria
        self.segment_list = []   # List of segments, including hits
        self.segment_mapping = {}   # Dictionary, name of the segment as key and list names of target segments as value

    def target_segments(self, name):
        """
        Takes the name of a segment and the target segments are returned
        :param name: name of segment
        :return: target segments
        """
        segment_names = [segment_name for segment_name in self.segment_mapping[name]]
        return [segment for segment in self.segment_list if segment.name in segment_names]

    def map_segments_simple_model(self):
        """Maps the segments according to the Preselection. Stores the result inside the segment mapping attribute."""

        # get maximum detector dimension in x
        x_dimension = []
        for key in self.selection_criteria["detector planes"].keys():
            x_dimension.append(self.selection_criteria["detector planes"][key]["x start"])
            x_dimension.append(self.selection_criteria["detector planes"][key]["x end"])
        min_x_detector_dimension = min(x_dimension)
        max_x_detector_dimension = max(x_dimension)

        for segment in self.segment_list:
            target_list = []
            for target_segment in self.segment_list:
                if target_segment.layer != segment.layer + 1:   # consecutive layers
                    continue
                if target_segment.x_end < segment.x_start:   # heavy scattering excluded
                    continue

                # max and minimum dx values
                max_dx = target_segment.x_end - segment.x_start
                min_dx = max([target_segment.x_start - segment.x_end, 0])

                # max x_0 range on reference screen
                x0_max = self.z_at_x0(target_segment.x_start,
                                      segment.x_end,
                                      target_segment.z_position,
                                      segment.z_position)

                x0_min = self.z_at_x0(target_segment.x_end,
                                      segment.x_start,
                                      target_segment.z_position,
                                      segment.z_position)

                # correct for detector dimensions
                if x0_min < min_x_detector_dimension:
                    x0_min = min_x_detector_dimension
                    if x0_max < min_x_detector_dimension:
                        continue
                if x0_max > max_x_detector_dimension:
                    x0_max = max_x_detector_dimension

                dx_x0_interval = pd.Interval(self.selection_criteria["doublet criteria"]["dx/x0 criteria"] -
                                             self.selection_criteria["doublet criteria"]["dx/x0 criteria epsilon"],
                                             self.selection_criteria["doublet criteria"]["dx/x0 criteria"] +
                                             self.selection_criteria["doublet criteria"]["dx/x0 criteria epsilon"])

                max_dx_interval = pd.Interval(min_dx / x0_max, max_dx / x0_min)

                if dx_x0_interval.overlaps(max_dx_interval):
                    target_list.append(target_segment.name)

            self.segment_mapping.update({segment.name: target_list})

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
        return x_end - dx * abs(z_end - self.selection_criteria["doublet criteria"]["reference layer z"]) / dz