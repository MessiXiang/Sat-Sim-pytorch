__all__ = ['SphericalHarmonicGravityBody']
import csv
import math
import torch
from .gravity_body import GravityBody
from torch import Tensor


class SphericalHarmonicGravityBody(GravityBody):

    def __init__(
        self,
        *args,
        gravity_file='/supportData/LocalGravData/GGM03S-J2-only.txt',
        max_degree=2,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.load_grav_from_file(gravity_file, max_deg=max_degree)
        self.initialize_parameters()

    def load_grav_from_file(self, file_name: str, max_deg: int = 2):

        [clm_list, slm_list, mu,
         rad_equator] = load_grav_from_file_to_list(file_name, max_deg=2)

        self.mu_body = mu
        self.rad_equator = rad_equator

        clm_list = clm_list[:max_deg + 1]
        slm_list = slm_list[:max_deg + 1]
        #Convert a lower triangular matrix into a full square matrix
        self.c_bar = lower_to_full_square(clm_list)
        self.s_bar = lower_to_full_square(slm_list)
        self.max_deg = max_deg

    def initialize_parameters(self):
        # initialize the parameters
        n1 = []
        n2 = []
        a_bar = []
        n_quot1 = []
        n_quot2 = []

        #init the basic parameters
        # calculate aBar / n1 / n2
        for i in range(self.max_deg + 2):
            a_row = [0.0] * (self.max_deg + 2)
            if i == 0:
                a_row[0] = 1.0
            else:
                a_row[i] = math.sqrt(
                    ((2 * i + 1) * get_k(i)) /
                    (2 * i * get_k(i - 1))) * a_bar[i - 1][i - 1]

            n1Row = [0.0] * (self.max_deg + 2)
            n2Row = [0.0] * (self.max_deg + 2)
            for m in range(i + 1):
                if i >= m + 2:
                    n1Row[m] = math.sqrt(
                        ((2 * i + 1) * (2 * i - 1)) / ((i - m) * (i + m)))
                    n2Row[m] = math.sqrt(
                        ((i + m - 1) * (2 * i + 1) *
                         (i - m - 1)) / ((i + m) * (i - m) * (2 * i - 3)))
            a_bar.append(a_row)
            n1.append(n1Row)
            n2.append(n2Row)

        # init nQuot1 / nQuot2
        for l in range(self.max_deg + 1):
            nq1_row = [0.0] * (self.max_deg + 1)
            nq2_row = [0.0] * (self.max_deg + 1)
            for m in range(l + 1):
                if m < l:
                    nq1_row[m] = math.sqrt(
                        ((l - m) * get_k(m) * (l + m + 1)) / get_k(m + 1))
                nq2_row[m] = math.sqrt(
                    ((l + m + 2) * (l + m + 1) *
                     (2 * l + 1) * get_k(m)) / ((2 * l + 3) * get_k(m + 1)))
            n_quot1.append(nq1_row)
            n_quot2.append(nq2_row)

        self.a_bar = a_bar
        self.n1 = n1
        self.n2 = n2
        self.n_quot1 = n_quot1
        self.n_quot2 = n_quot2

    def compute_gravitational_acceleration(
        self,
        relative_position: Tensor,
    ) -> Tensor:
        """
        Computes the gravitational acceleration for a celestial body using a spherical
        harmonics gravity model.

        Args:
            relative_position (Tensor): Position tensor with shape [num_positions, 3],
                representing the 3D position vectors of points relative to the body's
                reference frame (usually body-fixed or inertial frame).

        Returns:
            Tensor: Gravitational acceleration tensor with shape [num_positions, 3],
                representing the total gravitational acceleration at each position
                computed from the spherical harmonics expansion.

        Notes:
            - The gravitational potential is represented as a series expansion in
            spherical harmonics, using normalized coefficients C̄ₗₘ and S̄ₗₘ up to
            a specified maximum degree and order.
            - The acceleration is computed by taking the gradient of the potential,
            which includes central term and higher-order perturbation terms capturing
            the body's non-spherical mass distribution.
            - Coefficients are typically loaded from a gravity model file (e.g., EGM2008)
            and expressed in a body-fixed reference frame.
            - Compared to the point-mass model, this method captures effects such as
            oblateness (J₂ term), tesseral, and sectoral harmonics, improving accuracy
            for near-body orbital dynamics.
        """

        device = relative_position.device
        dtype = relative_position.dtype

        degree = self.max_deg
        include_zero_degree = True
        order = degree

        x = relative_position[..., 0].unsqueeze(-1)
        y = relative_position[..., 1].unsqueeze(-1)
        z = relative_position[..., 2].unsqueeze(-1)

        r = torch.norm(relative_position, dim=-1, keepdim=True)
        s = x / r
        t = y / r
        u = z / r

        a_bar_t = expand_matrix(self.a_bar,
                                relative_position,
                                dtype=dtype,
                                device=device)
        n1_t = expand_matrix(self.n1,
                             relative_position,
                             dtype=dtype,
                             device=device)
        n2_t = expand_matrix(self.n2,
                             relative_position,
                             dtype=dtype,
                             device=device)

        l_idx = torch.arange(1, degree + 2, device=a_bar_t.device)
        coef = torch.where(l_idx == 1, torch.ones_like(l_idx, dtype=u.dtype),
                           torch.sqrt(2.0 * l_idx.to(u.dtype)))
        a_bar_t[..., l_idx, l_idx - 1] = coef * a_bar_t[..., l_idx, l_idx] * u

        m_all = torch.arange(order + 2, device=a_bar_t.device)

        for l in range(2, degree + 2):
            limit = min(l - 1, order + 2)
            if limit <= 0:
                continue
            vm = m_all[:limit]

            a_bar_t[..., l,
                    vm] = (u * n1_t[..., l, vm] * a_bar_t[..., l - 1, vm] -
                           n2_t[..., l, vm] * a_bar_t[..., l - 2, vm])

        rE = torch.zeros(relative_position.shape[:-1] + (order + 2, ),
                         device=relative_position.device,
                         dtype=relative_position.dtype)

        iM = torch.zeros_like(rE)

        z_st = s + 1j * t
        m_idx = torch.arange(order + 2, device=z_st.device)
        z_pow = z_st**m_idx
        rE = z_pow.real
        iM = z_pow.imag

        rho = self.rad_equator / r
        rhol_0 = self.mu_body / r

        powers = torch.arange(degree + 2, device=rho.device, dtype=rho.dtype)
        rhol = rhol_0 * rho.pow(powers)

        a1 = torch.zeros_like(r)
        a2 = torch.zeros_like(r)
        a3 = torch.zeros_like(r)
        a4 = torch.zeros_like(r)
        if include_zero_degree:
            a4[..., 0] = -rhol[..., 1] / self.rad_equator

        sum_a1 = torch.zeros_like(r)
        sum_a2 = torch.zeros_like(r)
        sum_a3 = torch.zeros_like(r)
        sum_a4 = torch.zeros_like(r)

        c_bar_t = expand_matrix(self.c_bar,
                                relative_position,
                                dtype=dtype,
                                device=device)
        s_bar_t = expand_matrix(self.s_bar,
                                relative_position,
                                dtype=dtype,
                                device=device)
        n_quot1_t = expand_matrix(self.n_quot1,
                                  relative_position,
                                  dtype=dtype,
                                  device=device)
        n_quot2_t = expand_matrix(self.n_quot2,
                                  relative_position,
                                  dtype=dtype,
                                  device=device)

        for l in range(1, degree + 1):
            M = l + 1

            cL = c_bar_t[..., l, :M]
            sL = s_bar_t[..., l, :M]
            rE_m = rE[..., :M]
            iM_m = iM[..., :M]

            D = cL * rE_m + sL * iM_m

            rE_prev = torch.cat(
                [torch.zeros_like(rE_m[..., :1]), rE_m[..., :-1]], dim=-1)
            iM_prev = torch.cat(
                [torch.zeros_like(iM_m[..., :1]), iM_m[..., :-1]], dim=-1)

            E = cL * rE_prev + sL * iM_prev
            F = sL * rE_prev - cL * iM_prev

            m_idx = torch.arange(M,
                                 device=relative_position.device,
                                 dtype=relative_position.dtype)

            aL = a_bar_t[..., l, :M]
            aL_mplus1 = a_bar_t[..., l, 1:M]
            aLp1_mplus1 = a_bar_t[..., l + 1, 1:M + 1]

            nQ1 = n_quot1_t[..., l, :M - 1]
            nQ2 = n_quot2_t[..., l, :M]

            sum_a1 = (m_idx * aL * E).sum(dim=-1, keepdim=True)
            sum_a2 = (m_idx * aL * F).sum(dim=-1, keepdim=True)
            sum_a3 = (nQ1 * aL_mplus1 * D[..., :M - 1]).sum(dim=-1,
                                                            keepdim=True)
            sum_a4 = (nQ2 * aLp1_mplus1 * D).sum(dim=-1, keepdim=True)

            coeff = rhol[..., l + 1].unsqueeze(-1) / self.rad_equator
            a1 += coeff * sum_a1
            a2 += coeff * sum_a2
            a3 += coeff * sum_a3
            a4 -= coeff * sum_a4

        ax = a1 + s * a4
        ay = a2 + t * a4
        az = a3 + u * a4

        ax = ax.squeeze(-1)
        ay = ay.squeeze(-1)
        az = az.squeeze(-1)

        acceleration = torch.stack([ax, ay, az], dim=-1)

        return acceleration


def load_grav_from_file_to_list(file_name: str, max_deg: int = 2):
    with open(file_name, 'r') as csvfile:
        grav_reader = csv.reader(csvfile, delimiter=',')
        first_row = next(grav_reader)
        clm_list = []
        slm_list = []

        try:
            rad_equator = float(first_row[0])
            mu = float(first_row[1])
            max_degree_file = int(first_row[3])
            max_order_file = int(first_row[4])
            coefficients_normalized = int(first_row[5]) == 1
            ref_long = float(first_row[6])
            ref_lat = float(first_row[7])
        except Exception as ex:
            raise ValueError(
                "File is not in the expected JPL format for "
                "spherical Harmonics", ex)

        if max_degree_file < max_deg or max_order_file < max_deg:
            raise ValueError(
                f"Requested using Spherical Harmonics of degree {max_deg}"
                f", but file '{file_name}' has maximum degree/order of"
                f"{min(max_degree_file, max_order_file)}")

        if not coefficients_normalized:
            raise ValueError(
                "Coefficients in given file are not normalized. This is "
                "not currently supported in Basilisk.")

        if ref_long != 0 or ref_lat != 0:
            raise ValueError(
                "Coefficients in given file use a reference longitude"
                " or latitude that is not zero. This is not currently "
                "supported in Basilisk.")

        clm_row = []
        slm_row = []
        curr_deg = 0
        for grav_row in grav_reader:
            while int(grav_row[0]) > curr_deg:
                if (len(clm_row) < curr_deg + 1):
                    clm_row.extend([0.0] * (curr_deg + 1 - len(clm_row)))
                    slm_row.extend([0.0] * (curr_deg + 1 - len(slm_row)))
                clm_list.append(clm_row)
                slm_list.append(slm_row)
                clm_row = []
                slm_row = []
                curr_deg += 1
            clm_row.append(float(grav_row[2]))
            slm_row.append(float(grav_row[3]))

        return [clm_list, slm_list, mu, rad_equator]


def get_k(degree: int) -> float:
    return 1.0 if degree == 0 else 2.0


def expand_matrix(matrix_list, ref_tensor, dtype=None, device=None):
    batch, num = ref_tensor.shape[0], ref_tensor.shape[1]
    mat = torch.tensor(matrix_list, dtype=dtype, device=device)
    return mat.unsqueeze(0).unsqueeze(0).expand(batch, num, -1, -1).clone()


def lower_to_full_square(lower_list):
    """
    input: lower_list
    output: square_list
    """
    n = len(lower_list)
    full_matrix = []

    for i, row in enumerate(lower_list):
        new_row = row + [0.0] * (n - len(row))
        full_matrix.append(new_row)

    return full_matrix
