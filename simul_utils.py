import numpy as np


HUGEVAL = 1.0e30  # Valor enorme


class SimulationROI:
    """Classe que armazena os parâmetros da *Region of Interest* (ROI) para
    a simulação.


    Parameters
    ----------
        coord_ref : :class:`np.ndarray`
            Ponto cartesiano indicando a coordenada de referência da ROI, em
            mm. É referente ao ponto da ROI com índices (0, 0, 0), sem considerar as camadas e CPML.
            Por padrão, é (0.0, 0.0, 0.0) mm.

        height : int, float
            Altura da ROI, em mm. Corresponde ao eixo `z`. Por padrão, é 30.0 mm.

        h_len : int
            Quantidade de pontos na dimensão de altura ROI. Por padrão, é 300.

        width : int, float
            Largura da ROI (em um transdutor linear, tipicamente corresponde à
            direção ativa -- eixo `x`), em mm. Por padrão, é 30.0 mm.

        w_len : int
            Quantidade de pontos na dimensão de largura ROI. Por padrão, é 300.

        depth : int, float
            Profundidade da ROI (em um transdutor linear, tipicamente corresponde à
            direção passiva -- eixo `y`), em mm. Por padrão, é 0.0 mm (ROI de duas dimensões).

        d_len : int
            Quantidade de pontos na dimensão de profundidade ROI. Por padrão, é 1
            (ROI de duas dimensões).

        pad : int
            Quantidade de pontos adicionais em cada lado da dimensao da ROI. Por padrão é 1.

    Attributes
    ----------
        coord_ref : :class:`np.ndarray`
            Ponto cartesiano indicando a coordenada de referência da ROI, em mm.

        h_points : :class:`np.ndarray`
            Vetor com as coordenadas da ROI no sentido da altura (dimensão 3 -- eixo `z`)
            da ROI de simulação, em mm.

        h_len : int
            Quantidade de pontos da ROI no sentido da altura.

        h_step : float
            Tamanho do passo dos pontos da ROI no sentido da altura, em mm.

        height : float
            Altura da ROI, em mm.

        w_points : :class:`np.ndarray`
            Vetor com as coordenadas da ROI no sentido da largura (dimensão 1 -- eixo `x`)
            da ROI de simulação.

        w_len : int
            Quantidade de pontos da ROI no sentido da largura.

        w_step : float
            Tamanho do passo dos pontos da ROI no sentido da largura, em mm.

        width : float
            Largura da ROI, em mm.

        d_points : :class:`np.ndarray`
            Vetor com as coordenadas da ROI no sentido da profundidade (dimensão 2 -- eixo `y`)
            da ROI de simulação.

        d_len : int
            Quantidade de pontos da ROI no sentido da profundidade.

        d_step : float
            Tamanho do passo dos pontos da ROI no sentido da profundidade, em mm.

        depth : float
            Profundidade da ROI, em mm.

        pad : int
            Quantidade adicional de pontos em cada lado das dimensões da ROI.

    Raises
    ------
    TypeError
        Gera exceção de ``TypeError`` se ``coord_ref`` não for do tipo
        :class:`np.ndarray` e/ou não possuir 1 linha e três colunas.

    Notes
    -----
    Esta classe aplica-se a ROIs em duas e três dimensões.

    """

    def __init__(self, coord_ref=np.zeros((1, 3)), height=30.0, h_len=300, width=30.0, w_len=300, depth=0.0, d_len=1,
                 len_pml_xmin=10, len_pml_xmax=10, len_pml_ymin=10, len_pml_ymax=10, len_pml_zmin=10, len_pml_zmax=10,
                 pad=1):
        if type(coord_ref) is list:
            coord_ref = np.array(coord_ref)

        if (coord_ref is not np.ndarray) and (coord_ref.shape != (1, 3)):
            raise TypeError("``coord_ref`` deve ser um vetor-linha de 3 elementos [shape = (1,3)]")

        # Atribuicao dos atributos da instancia.
        # Ponto cartesiano indicando a coordenada de referencia da ROI.
        self.coord_ref = coord_ref

        # Vetor com as coordenadas da ROI no sentido da altura (dimensao 3 - eixo 'z') da simulacao.
        self.h_points = np.linspace(coord_ref[0, 2], coord_ref[0, 2] + height, num=int(h_len), endpoint=False)

        # Quantidade de pontos da ROI no sentido da altura.
        self._h_len = self.h_points.size

        # Passo dos pontos da ROI no sentido da altura.
        self.h_step = height / h_len  # self.h_points[1] - self.h_points[0]

        # Altura da ROI.
        self.height = self.h_points[-1] + self.h_points[1] - 2 * self.h_points[0]

        # Vetor com as coordenadas da ROI no sentido da largura (dimensão 1 - eixo 'x') da simulacao.
        self.w_points = np.linspace(coord_ref[0, 0], coord_ref[0, 0] + width, num=int(w_len), endpoint=False)

        # Quantidade de pontos da ROI no sentido da largura.
        self._w_len = self.w_points.size

        # Passo dos pontos da ROI no sentido da largura.
        self.w_step = width / w_len  # self.w_points[1] - self.w_points[0]

        # Largura da ROI.
        self.width = width  # self.w_points[-1] + self.w_points[1] - 2 * self.w_points[0]

        # Vetor com as coordenadas da ROI no sentido da profundidade (dimensão 2 - eixo 'y') da simulacao.
        self.d_points = np.linspace(coord_ref[0, 1], coord_ref[0, 1] + depth, num=int(d_len), endpoint=False)

        # Quantidade de pontos da ROI no sentido da profundidade.
        self._d_len = self.d_points.size

        # Profundidade da ROI.
        self.depth = depth

        # Passo dos pontos da ROI no sentido da profundidade.
        if d_len > 1:
            self.d_step = self.d_points[1] - self.d_points[0]
        else:
            self.d_step = self.w_step

        # Tamanho das camadas de PML, 0 se não for para calcular
        self._pml_xmin_len = len_pml_xmin
        self._pml_xmax_len = len_pml_xmax
        self._pml_ymin_len = len_pml_ymin
        self._pml_ymax_len = len_pml_ymax
        self._pml_zmin_len = len_pml_zmin
        self._pml_zmax_len = len_pml_zmax

        # Quantidade adicional de pontos em cada lado da ROI.
        self._pad = pad

        # Vetores com os valores dos coeficientes para cálculo da PML
        if self._pml_xmin_len + self._pml_xmax_len > 0:
            self._pml_full_x = np.zeros((self._pml_xmin_len + self._pml_xmax_len + self._w_len, 3))
            self._pml_half_x = np.zeros((self._pml_xmin_len + self._pml_xmax_len + self._w_len, 3))

        if self._pml_ymin_len + self._pml_ymax_len > 0:
            self._pml_full_y = np.zeros((self._pml_ymin_len + self._pml_ymax_len + self._d_len, 3))
            self._pml_half_y = np.zeros((self._pml_ymin_len + self._pml_ymax_len + self._d_len, 3))

        if self._pml_zmin_len + self._pml_zmax_len > 0:
            self._pml_full_z = np.zeros((self._pml_zmin_len + self._pml_zmax_len + self._h_len, 3))
            self._pml_half_z = np.zeros((self._pml_zmin_len + self._pml_zmax_len + self._h_len, 3))

    def get_nx(self):
        return self._w_len + self._pml_xmin_len + self._pml_xmax_len + 2 * self._pad

    def get_ny(self):
        return self._d_len + self._pml_ymin_len + self._pml_ymax_len + 2 * self._pad

    def get_nz(self):
        return self._h_len + self._pml_zmin_len + self._pml_zmax_len + 2 * self._pad

    def get_len_x(self):
        return self._w_len

    def get_len_y(self):
        return self._d_len

    def get_len_z(self):
        return self._h_len

    def get_ix_min(self):
        return self._pml_xmin_len + self._pad

    def get_ix_max(self):
        return self._w_len + self._pml_xmin_len + self._pad

    def get_iy_min(self):
        return self._pml_ymin_len + self._pad

    def get_iy_max(self):
        return self._d_len + self._pml_ymin_len + self._pad

    def get_iz_min(self):
        return self._pml_zmin_len + self._pad

    def get_iz_max(self):
        return self._h_len + self._pml_zmin_len + self._pad

    def get_pml_thickness_x(self):
        return (self._pml_xmin_len + self._pml_xmax_len) * self.w_step

    def get_pml_thickness_y(self):
        return (self._pml_ymin_len + self._pml_ymax_len) * self.d_step

    def get_pml_thickness_z(self):
        return (self._pml_zmin_len + self._pml_zmax_len) * self.h_step

    def get_nearest_grid_idx(self, point):
        """Método para retornar os índices mais próximos da grade para o ponto da ROI fornecido."""
        if type(point) is not np.ndarray and type(point) is list:
            point = np.ndarray(point)

        if not (self.w_points[0] <= point[0] <= self.w_points[-1]):
            raise IndexError(f"'x' = {point[0]} out of bounds")

        if not (self.d_points[0] <= point[1] <= self.d_points[-1]):
            raise IndexError(f"'y' = {point[1]} out of bounds")

        if not (self.h_points[0] <= point[2] <= self.h_points[-1]):
            raise IndexError(f"'z' = {point[2]} out of bounds")

        ix = np.absolute(self.w_points - point[0]).argmin() + self._pml_xmin_len + self._pad
        iy = np.absolute(self.d_points - point[1]).argmin() + self._pml_ymin_len + self._pad
        iz = np.absolute(self.h_points - point[2]).argmin() + self._pml_zmin_len + self._pad
        return [ix, iy, iz]

    def calc_pml_array(self, axis='x', grid='f', dt=1.0, d0=1.0, npower=2.0, alpha_max=30.0, k_max=1.0):
        """Metodo que calcula os vetores com os valores para implementar a camada de PML"""
        # Origem da PML (posicao das bordas direita e esquerda menos a espessura, em unidades de distancia)
        if axis == 'x' or axis == 'X':
            delta = self.w_step
            tam_pml = self._w_len + self._pml_xmin_len + self._pml_xmax_len
            orig_left = self._pml_xmin_len * delta
            orig_right = (self._pml_xmin_len + self._w_len - 1) * delta
            thickness_pml_left = self._pml_xmin_len * delta
            thickness_pml_right = self._pml_xmax_len * delta
        elif axis == 'y' or axis == 'Y':
            delta = self.d_step
            tam_pml = self._d_len + self._pml_ymin_len + self._pml_ymax_len
            orig_left = self._pml_ymin_len * delta
            orig_right = (self._pml_ymin_len + self._d_len - 1) * delta
            thickness_pml_left = self._pml_ymin_len * delta
            thickness_pml_right = self._pml_ymax_len * delta
        elif axis == 'z' or axis == 'Z':
            delta = self.h_step
            tam_pml = self._h_len + self._pml_zmin_len + self._pml_zmax_len
            orig_left = self._pml_zmin_len * delta
            orig_right = (self._pml_zmin_len + self._h_len - 1) * delta
            thickness_pml_left = self._pml_zmin_len * delta
            thickness_pml_right = self._pml_zmax_len * delta
        else:
            raise IndexError(f"'axis' = {axis} not supported")

        # Incicializacao para full ou half grid
        val = delta * np.arange(tam_pml)
        if grid == 'f' or grid == 'F':
            val_pml_left = orig_left - val
            val_pml_right = val - orig_right
        elif grid == 'h' or grid == 'H':
            val_pml_left = orig_left - (val + delta / 2.0)
            val_pml_right = (val + delta / 2.0) - orig_right
        else:
            raise IndexError(f"'grid' = {grid} not supported")

        # Calculo dos coeficientes
        pml_mask_left = np.where(val_pml_left < 0.0, False, True)
        pml_mask_right = np.where(val_pml_right < 0.0, False, True)
        mask = np.logical_or(pml_mask_left, pml_mask_right)
        pml = np.zeros(tam_pml)
        pml[pml_mask_left] = val_pml_left[pml_mask_left] / thickness_pml_left
        pml[pml_mask_right] = val_pml_right[pml_mask_right] / thickness_pml_right
        d = (d0 * pml ** npower).astype(np.float32)
        k = (1.0 + (k_max - 1.0) * pml ** npower).astype(np.float32)
        alpha = (alpha_max * (1.0 - np.where(mask, pml, 1.0))).astype(np.float32)
        b = np.exp(-(d / k + alpha) * dt).astype(np.float32)
        a = np.zeros(tam_pml, dtype=np.float32)
        i = np.where(d > 1e-6)
        a[i] = d[i] * (b[i] - 1.0) / (k[i] * (d[i] + k[i] * alpha[i]))

        return a, b, k

    # def get_pml_arrays(self):
    #     """Metodo que calcula e retorna os vetores como os valores para implementar a camada de PML"""

    # def get_coord(self):
    #     """Método que retorna todas as coordenadas da ROI (*mesh*) no formato
    #     vetorizado.
    #
    #     Returns
    #     -------
    #         : :class:`np.ndarray`
    #             Matriz :math:`M` x 3, em que :math:`M` é a quantidade de
    #             pontos existentes na ROI. Cada linha dessa matriz é a
    #             coordenada cartesiana de um ponto da ROI.
    #
    #     """
    #     return np.array(np.meshgrid(self.w_points,
    #                                 self.d_points,
    #                                 self.h_points,
    #                                 indexing='ij')).reshape((3, -1)).T
