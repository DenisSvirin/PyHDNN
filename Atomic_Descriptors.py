import numpy as np
import pandas as pd
from tqdm import tqdm

class BaseFunctions:
    """
    Base class for creating atomic description
    """

    def __init__(self, cutoff_R=6.5, a=3.615):

        self.cutoff_R = cutoff_R
        self.a = a

    def f(self, R):
        # defines cutoff sphere
        # We use only R s that are within cutoff_R
        if R > self.cutoff_R:
            return 0.0
        else:
            return 0.5 * (np.cos(np.pi * R / self.cutoff_R) + 1)

    def distance(self, r_1, r_2):
        # Calculates distance between r_1, r_2
        # r = [x, y, z]
        return np.sqrt(
            (r_2[0] - r_1[0]) ** 2 + (r_2[1] - r_1[1]) ** 2 + (r_2[2] - r_1[2]) ** 2
        )

    def count_non_zero_values(self, coordinates):
        # Counting none zero values in r
        # r = [x, y, z]
        return (coordinates != 0).sum()

    def _add(self, dictionary, key, value):
        # add key - value to dictionary
        if str(key) in dictionary.keys():
            dictionary[str(key)] += 2 ** self.count_non_zero_values(value)

        else:
            dictionary[str(key)] = 2 ** self.count_non_zero_values(value)

    def get_distance_dict(self, atomic_coordinates, scale_factor=1, r_cutoff=1.4):
        # returns dictionary of distances between point [0, 0, 0]
        # and all other points in atomic_coordinates within r_cutoff
        distance_dict = dict()
        for atom_i in atomic_coordinates:
            current_atom = atom_i
            distance_i_j = self.distance([0, 0, 0], current_atom)
            scaled_distance = scale_factor * distance_i_j
            if scaled_distance == 0:
                continue

            if scaled_distance < r_cutoff:
                self._add(distance_dict, scaled_distance, current_atom)

        return distance_dict

    def generate_self_centered_eta(self, N):

        return [[0, (N ** (m / N) / self.cutoff_R) ** 2] for m in range(N)]

    def generate_R(self, n, m):
        return self.cutoff_R / n ** (m / n)

    def generate_eta(self, N):
        return [
            [
                self.generate_R(N, m),
                1 / (self.generate_R(N, N - m) - self.generate_R(N, N - m - 1)) ** 2,
            ]
            for m in range(N)
        ]

    def _get_G_1(self, distances, n, R_c):
        # returns values of G_1 (radial) fuction for given
        # distances (dict [distance] -> amount of atoms)
        # n - responsable for width of the curve
        # R_c - responsable for horizontal position of the curve
        G_1_i = 0
        for i in distances.keys():
            R_i_j = float(i)
            amount_of_atoms = distances[i]
            G_1_i += amount_of_atoms * np.exp(-n * (R_i_j - R_c) ** 2) * self.f(R_i_j)
        return G_1_i

    def _get_G_2(self, coordinates, c, n):

        # coordinates - cartesian coordinates of all
        # atoms in x >= 0, y >= 0, z >=0 area

        G_2_i_p = 0
        G_2_i_n = 0

        for j in coordinates[1:]:
            for k in coordinates[1:]:
                if np.array_equal(j, k):
                    continue
                else:
                    R_i_j = self.distance([0, 0, 0], j)
                    R_i_k = self.distance([0, 0, 0], k)
                    R_j_k = self.distance(j, k)
                    if (
                        R_i_j >= self.cutoff_R
                        or R_i_k >= self.cutoff_R
                        or R_j_k >= self.cutoff_R
                    ):
                        continue
                    cos_t = j @ k / (R_i_j * R_i_k)

                    atoms_ = 2 ** self.count_non_zero_values(
                        j
                    ) + 2 ** self.count_non_zero_values(k)
                    G_2_i_p += (
                        atoms_
                        * ((1 + cos_t) ** c)
                        * np.exp(-n * (R_i_j**2 + R_i_k**2 + R_j_k**2))
                        * self.f(R_i_j)
                        * self.f(R_i_j)
                        * self.f(R_j_k)
                    )

                    G_2_i_n += (
                        atoms_
                        * ((1 - cos_t) ** c)
                        * np.exp(-n * (R_i_j**2 + R_i_k**2 + R_j_k**2))
                        * self.f(R_i_j)
                        * self.f(R_i_j)
                        * self.f(R_j_k)
                    )

        G_2_i_p /= 2  # because we included j,k and k, j
        G_2_i_n /= 2  # because we included j,k and k, j

        return (2.0 ** (1 - c)) * G_2_i_p, (2.0 ** (1 - c)) * G_2_i_n


class SymmetryFunctions(BaseFunctions):
    """
    Class to create 2 sets:
    1) set of G1 functions  (length = len_g1_functions)
    2) set of G2 functions  (length = len_g2_functions)
    """

    def __init__(
        self,
        cutoff_R=6.5,
        zetas=[],
        len_g1_functions=8,
        len_g2_functions=21,
        lattice_constants=[],
        atomic_coordinates=[],
        no_angles=False,
    ):
        super().__init__(cutoff_R=cutoff_R)
        self.symmetry_functions = []
        self.symmetry_functions_dataframe = None
        self.len_g1_functions = len_g1_functions
        self.len_g2_functions = len_g2_functions
        self.lattice_constants = lattice_constants
        self.atomic_coordinates = atomic_coordinates
        self.no_angles = no_angles
        self.zetas = zetas
        if len(self.zetas) == 0:
            self.zetas = np.random.choice([1, 2, 4, 16], self.len_g2_functions)

        assert (
            len(self.zetas) == len_g2_functions
        ), "Custom zeta set should be length equal to len_g2_functions"

    def transform_to_SymmetryFunctions(self):
        """
        Creates set of symmetry function for a given
        structure and lattice constant
        """
        l_g1 = self.len_g1_functions
        l_g2 = self.len_g2_functions
        R_and_n_for_G1 = self.generate_self_centered_eta(N=l_g1) + self.generate_eta(
            N=l_g1
        )
        R_and_n_for_G2 = self.generate_self_centered_eta(N=self.len_g2_functions)
        for _r in tqdm(self.lattice_constants, desc="Converted"):
            distances = self.get_distance_dict(
                atomic_coordinates=self.atomic_coordinates,
                r_cutoff=self.cutoff_R,
                scale_factor=_r,
            )
            G1_functions = [
                self._get_G_1(distances, R_c=_params[0], n=_params[1])
                for _params in R_and_n_for_G1
            ]
            G2_functions = []

            for j, c in enumerate(self.zetas):
                if self.no_angles:
                    G2_functions.extend([0.0, 0.0])
                else:
                    _G_2 = self._get_G_2(
                        coordinates=self.atomic_coordinates * _r,
                        c=c,
                        n=R_and_n_for_G2[j][1],
                    )
                    G2_functions.extend(_G_2)

            combined_G1_G2 = np.hstack((G1_functions, G2_functions))
            self.symmetry_functions.extend([np.array(combined_G1_G2)])
        self.symmetry_functions_dataframe = pd.DataFrame(
            self.symmetry_functions,
            columns=["feature_" + str(i) for i in range((l_g1 + l_g2) * 2)],
        )
