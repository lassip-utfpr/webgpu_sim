import numpy as np
from scipy.signal import gausspulse

HUGEVAL = 1.0e30  # Valor enorme


class SimulationROI:
    """
    Classe que armazena os parâmetros da *Region of Interest* (ROI) para
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

        _h_len : int
            Quantidade de pontos da ROI no sentido da altura.

        h_step : float
            Tamanho do passo dos pontos da ROI no sentido da altura, em mm.

        height : float
            Altura da ROI, em mm.

        w_points : :class:`np.ndarray`
            Vetor com as coordenadas da ROI no sentido da largura (dimensão 1 -- eixo `x`)
            da ROI de simulação.

        _w_len : int
            Quantidade de pontos da ROI no sentido da largura.

        w_step : float
            Tamanho do passo dos pontos da ROI no sentido da largura, em mm.

        width : float
            Largura da ROI, em mm.

        d_points : :class:`np.ndarray`
            Vetor com as coordenadas da ROI no sentido da profundidade (dimensão 2 -- eixo `y`)
            da ROI de simulação.

        _d_len : int
            Quantidade de pontos da ROI no sentido da profundidade.

        d_step : float
            Tamanho do passo dos pontos da ROI no sentido da profundidade, em mm.

        depth : float
            Profundidade da ROI, em mm.

        _pad : int
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

        # Passo dos pontos da ROI no sentido da altura.
        self.h_step = height / h_len

        # Vetor com as coordenadas da ROI no sentido da altura (dimensao 3 - eixo 'z') da simulacao.
        self.dec_h = int(abs(np.log10(self.h_step))) + 1
        self.h_points = np.linspace(coord_ref[0, 2], coord_ref[0, 2] + height, num=int(h_len), endpoint=False,
                                    dtype=np.float32).round(decimals=self.dec_h)

        # Quantidade de pontos da ROI no sentido da altura.
        self._h_len = self.h_points.size

        # Altura da ROI.
        self.height = height

        # Passo dos pontos da ROI no sentido da largura.
        self.w_step = width / w_len

        # Vetor com as coordenadas da ROI no sentido da largura (dimensão 1 - eixo 'x') da simulacao.
        self.dec_w = int(abs(np.log10(self.w_step))) + 1
        self.w_points = np.linspace(coord_ref[0, 0], coord_ref[0, 0] + width, num=int(w_len), endpoint=False,
                                    dtype=np.float32).round(decimals=self.dec_w)

        # Quantidade de pontos da ROI no sentido da largura.
        self._w_len = self.w_points.size

        # Largura da ROI.
        self.width = width

        # Passo dos pontos da ROI no sentido da profundidade.
        if depth > 0.0 and d_len > 1:
            self.d_step = depth / d_len
        else:
            self.d_step = self.w_step

        # Vetor com as coordenadas da ROI no sentido da profundidade (dimensão 2 - eixo 'y') da simulacao.
        self.dec_d = int(abs(np.log10(self.d_step))) + 1
        self.d_points = np.linspace(coord_ref[0, 1], coord_ref[0, 1] + depth, num=int(d_len), endpoint=False,
                                    dtype=np.float32).round(decimals=self.dec_d)

        # Quantidade de pontos da ROI no sentido da profundidade.
        self._d_len = self.d_points.size

        # Profundidade da ROI.
        self.depth = depth

        # Tamanho das camadas de PML, 0 se não for para calcular
        self._pml_xmin_len = len_pml_xmin
        self._pml_xmax_len = len_pml_xmax
        self._pml_ymin_len = len_pml_ymin
        self._pml_ymax_len = len_pml_ymax
        self._pml_zmin_len = len_pml_zmin
        self._pml_zmax_len = len_pml_zmax

        # Quantidade adicional de pontos em cada lado da ROI.
        self._pad = pad

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

        ix = np.absolute(self.w_points - np.round(point[0] - self.w_step / 10.0 ** (self.dec_w - 1),
                                                  decimals=self.dec_w)).argmin() + self._pml_xmin_len + self._pad
        iy = np.absolute(self.d_points - np.round(point[1] - self.d_step / 10.0 ** (self.dec_d - 1),
                                                  decimals=self.dec_d)).argmin() + self._pml_ymin_len + self._pad
        iz = np.absolute(self.h_points - np.round(point[2] - self.h_step / 10.0 ** (self.dec_h - 1),
                                                  decimals=self.dec_h)).argmin() + self._pml_zmin_len + self._pad
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


class SimulationProbe:
    """
    Classe base contendo as configurações do transdutor para a simulação.

    Na implementação atual, os tipos suportados são mono e linear.

    Parameters
    ----------
        coord_center : :class:`np.ndarray`
            Coordenada relativa ao centro geométrico do transdutor. Por padrão se assume os
            valores [0.0, 0.0, 0.0].

    Attributes
    ----------
        coord_center : :class:`np.ndarray`
            Coordenada relativa ao centro geométrico do transdutor.

    """

    def __init__(self, coord_center=np.zeros((1, 3))):
        # Coordenada central do transdutor, em relacao ao ponto de referencia da ROI.
        self.coord_center = (coord_center if coord_center is np.ndarray else np.array(coord_center)).astype(np.float32)


class ElementRect:
    """
    Classe que define um elemento retangular de um transdutor de ultrassom.
    Essa classe pode ser utilizada nos transdutores "LinearArray" e "MonoRect".
    """

    def __init__(self, dim_a=0.5, dim_p=10.0, coord_center=np.zeros((1, 3)),
                 freq=5., bw=0.5, gain=1.0, t0=1.0, pulse_type="gaussian"):
        # Dimensao no sentido ativo do transdutor.
        self.elem_dim_a = np.float32(dim_a)

        # Dimensao no sentido passivo do transdutor.
        self.elem_dim_p = np.float32(dim_p)

        # Coordenada central do elemento
        self.coord_center = coord_center.astype(np.float32)

        # Parametros referentes ao sinal de excitacao do transdutor.
        # Frequencia central, em MHz.
        self.freq = np.float32(freq)

        # Banda passante, em percentual da frequencia central.
        self.bw = np.float32(bw)

        # Ganho do elemento transdutor.
        self.gain = np.float32(gain)

        # Atraso do sinal de excitacao.
        self.t0 = np.float32(t0)

        # Tipo do pulso de excitacao. O unico tipo possivel e: ``gaussian``.
        self.pulse_type = pulse_type

    def get_element_exc_fn(self, t):
        return self.gain * np.float32(gausspulse((t - self.t0), fc=self.freq, bw=self.bw))

    def get_num_points_roi(self, sim_roi=SimulationROI(), simul_type="2D"):
        """
        Metodo que retorna o número de os pontos ativos do transdutor no grid de simulacao.

        Returns
        -------
            : int
                Quantidade de pontos ativos (fontes) do elemento transdutor.

        """
        dim_p = min(self.elem_dim_p, sim_roi.depth)
        num_pt_a = int(np.round(self.elem_dim_a / sim_roi.w_step, decimals=sim_roi.dec_w))
        num_pt_p = int(np.round(dim_p / sim_roi.d_step, decimals=sim_roi.dec_d)) if dim_p != 0.0 else 1
        simul_type = simul_type.lower()
        num_coord = num_pt_a
        if simul_type == "3d":
            num_coord *= num_pt_p

        return num_coord

    def get_points_roi(self, sim_roi=SimulationROI(), simul_type="2D"):
        """
        Metodo que retorna as coordenadas de todas os pontos ativos do transdutor no grid de simulacao,
        no formato vetorizado.

        Returns
        -------
            : :class:`np.ndarray`
                Matriz :math:`M` x 3, em que :math:`M` é a quantidade de
                pontos ativos (fontes) do elemento transdutor como indices de pontos na ROI.
                Cada linha dessa matriz e o indice 3D de um ponto na ROI.

        """
        dim_p = min(self.elem_dim_p, sim_roi.depth)
        num_pt_a = int(np.round(self.elem_dim_a / sim_roi.w_step, decimals=sim_roi.dec_w))
        num_pt_p = int(np.round(dim_p / sim_roi.d_step, decimals=sim_roi.dec_d)) if dim_p != 0.0 else 1
        simul_type = simul_type.lower()
        num_coord = num_pt_a
        if simul_type == "3d":
            num_coord *= num_pt_p

        list_out = list()
        for p in range(num_coord):
            x_coord = np.float32((p % num_pt_a) * sim_roi.w_step - self.elem_dim_a / 2.0)
            y_coord = np.float32(0.0 if simul_type == "2d"
                                 else ((p // num_pt_a) % num_pt_p) * sim_roi.d_step - dim_p / 2.0)
            list_out.append([np.round(x_coord + self.coord_center[0], decimals=sim_roi.dec_w),
                             np.round(y_coord + self.coord_center[1], decimals=sim_roi.dec_d),
                             np.round(self.coord_center[2], decimals=sim_roi.dec_h)])

        return list_out


class SimulationProbeLinearArray(SimulationProbe):
    """
    Classe contendo as configurações de um transdutor do tipo array linear.
    É uma classe derivada de ``SimulationProbe``, específica para os transdutore do tipo
    "LinearArray".

    Parameters
    ----------
        coord_center : :class:`np.ndarray`
            Coordenada relativa ao centro geométrico do transdutor. Por padrão se assume os
            valores [0.0, 0.0, 0.0].

        num_elem : int
            Número de elementos. Exclusivo para transdutores do tipo
            ``linear``. Por padrão, é 32.

        dim_a : int, float
            Dimensão na direção ativa elementos do transdutor, em mm. Por padrão é 0.5 mm.

        inter_elem : int, float
            Espaçamento entre elementos, em mm. Por padrão é 0.1 mm.

        freq : int, float
            Frequência central, em MHz. Por padrão, é 5 MHz.

        bw : int, float
            Banda passante, em percentual da frequência central. Por padrão,
            é 0.5 (50%).

        pulse_type : str
            Tipo do pulso de excitação. Os tipos possíveis são: ``gaussian``,
            ``cossquare``, ``hanning`` e ``hamming``. Por padrão, é
            ``gaussian``.

    Attributes
    ----------
        num_elem : int
            Número de elementos. Exclusivo para transdutores da classe *array*.

        inter_elem : int, float
            Espaçamento entre elementos, em mm. Exclusivo para transdutores da
            classe *array*.

        pitch: int, float
            Espaçamento entre os centros dos elementos, em mm. Exclusivo para transdutores da
            classe *array*.

        elem_list : :class:`ElementRect`
            É uma lista de objetos do tipo ``ElementRect``, contendo as caracteristicas
            físicas e elétricas dos elementos ativos do transdutor.

    """

    def __init__(self, coord_center=np.zeros((1, 3)), num_elem=32, dim_a=0.5, dim_p=10.0, inter_elem=0.1,
                 freq=5., bw=0.5, gain=1.0, pulse_type="gaussian", t0_emmition=np.ones(32)):
        # Chama o construtor da classe base.
        super().__init__(coord_center)

        # Espacamento entre elementos.
        self.inter_elem = np.float32(inter_elem)

        # Tempo de atraso para emissao dos elementos. Se for um valor escalar, e assumido para todos os elementos.
        # Se for um array, deve ter um valor para cada elemento.
        if type(t0_emmition) is list:
            if len(t0_emmition) < num_elem:
                t0_emmition += [1.0 for _ in range(num_elem - len(t0_emmition))]
            elif len(t0_emmition) > num_elem:
                t0_emmition = t0_emmition[:num_elem]

            t0_emmition = np.array(t0_emmition, dtype=np.float32)
        elif type(t0_emmition) is np.array:
            if t0_emmition.shape[0] < num_elem:
                pad = np.ones((num_elem - t0_emmition.shape[0]), dtype=np.float32)
                t0_emmition = np.concatenate((t0_emmition, pad))
            elif t0_emmition.shape[0] > num_elem:
                t0_emmition = t0_emmition[:num_elem]
        elif type(t0_emmition) is float:
            t0_emmition = np.ones(num_elem, dtype=np.float32) * t0_emmition

        self.t0_emmition = t0_emmition.astype(dtype=np.float32)

        # Espacamento entre os centros dos elementos.
        self.pitch = np.float32(dim_a + inter_elem)

        # Numero de elementos.
        self.num_elem = num_elem
        offset_center = np.array([((num_elem - 1) * self.pitch + dim_a) / 2.0, 0.0, 0.0], dtype=np.float32)
        self.elem_list = [ElementRect(dim_a=dim_a, dim_p=dim_p,
                                      coord_center=np.array([dim_a / 2.0 + i * self.pitch, 0.0, 0.0],
                                                            dtype=np.float32) - offset_center,
                                      freq=freq, bw=bw, gain=gain,pulse_type=pulse_type,
                                      t0=float(self.t0_emmition[i]))
                          for i in range(num_elem)]

        # Parametros geometricos gerais do transdutor
        self._dim_a = dim_a
        self._dim_p = dim_p

        # Parametros eletricos gerais do transdutor
        self._freq = freq
        self._bw = bw
        self._gain = gain
        self._pulse_type = pulse_type

        # Transdutores matriciais
        #     # Numero de elementos.
        #     self.num_elem = num_elem
        #
        #     # Posicao central do elemento ativo do transdutor.
        #     self.elem_center = np.zeros((num_elem, 3))
        #
        #     # Espacamento entre elementos.
        #     self.inter_elem = inter_elem
        #
        # Transdutores circulares
        #     # Numero de elementos.
        #     self.num_elem = num_elem
        #
        #     # Posicao central do elemento ativo do transdutor.
        #     self.elem_center = np.zeros((num_elem, 3))
        #
        #     # Espacamento entre elementos.
        #     self.inter_elem = inter_elem
        #
        #     self.elem_list = elem_list
        #
        # Transdutores ``mono``.
        # # Formato do transdutor. Os valores possíveis sao ``circle``
        # # e ``rectangle``. O valor padrao é ``circle``.
        # self.shape = "circle"
        #
        # # Posicao central do elemento ativo do transdutor.
        # self.elem_center = np.zeros((num_elem, 3))
        #
        # # Espacamento entre elementos.
        # self.inter_elem = inter_elem
        #
        # # Espacamento entre os centros dos elementos.
        # self.pitch = dim_a + inter_elem

    def get_freq(self, mode='common'):
        """
        Método que retorna a frequência do transdutor.
        :param mode: str
            Este parâmetro define o modo de obtenção da frequência.
            "common" significa que será utilizado o parâmetro geral, utilizado por todos os elementos ativos.
            Este é o padrão.
            "mean" significa que será calculada a média das frequências de cada elemeto ativo do transdutor.
            "max" significa a maior frequência entre todos os elementos ativos do transdutor.

        :return: numpy.float32
            Retorna o valor da frequência do transdutor.
        """
        if mode == 'mean':
            return np.array([e.freq for e in self.elem_list]).mean()
        elif mode == 'max':
            return np.array([e.freq for e in self.elem_list]).max()
        else:
            return self._freq

    def get_points_roi(self, sim_roi=SimulationROI(), simul_type="2D"):
        """Metodo que retorna as coordenadas de todos os pontos ativos do transdutor no grid de simulacao,
        no formato vetorizado.

        Returns
        -------
            : :class:`np.ndarray`
                Matriz :math:`M` x 3, em que :math:`M` é a quantidade de
                pontos ativos (fontes) do transdutor como indices de pontos na ROI. Cada linha dessa matriz e a
                coordenada cartesiana (como indice) de um ponto na ROI.

        """
        arr_elem = list()
        for e in self.elem_list:
            arr_elem += e.get_points_roi(sim_roi=sim_roi, simul_type=simul_type)

        arr_out = [sim_roi.get_nearest_grid_idx(p + self.coord_center) for p in arr_elem]
        return np.array(arr_out).reshape(-1, 3)

    def get_source_term(self, samples=1000, dt=1.0, sim_roi=SimulationROI(), simul_type="2D"):
        """
        Metodo que retorna os sinais dos termos de fonte do transdutor. Além de retornar um
        array com os sinais dos termos de fonte de cada elemento ativo do transdutor, esta função
        também retorna um array com o índice do termo de fonte para cada ponto da ROI que é
        um ponto emissor.
        :param simul_type:
        :param sim_roi:
        :param samples: int
            Número de amostras de tempo na simulação.
        :param dt: float
            Valor do passo de tempo na simulação.

        :return: :numpy.array, :numpy.array
        O primeiro array tem dimensões de N amostras de tempo (linhas) por M elementos do transdutor (colunas).
        O segundo array tem dimensão única com a quantidade total de pontos emissores na ROI.
        """
        t = np.arange(samples, dtype=np.float32) * dt
        source_term = np.zeros((samples, self.num_elem), dtype=np.float32)
        idx_src = list()
        for idx_st, e in enumerate(self.elem_list):
            source_term[:, idx_st] = e.get_element_exc_fn(t)
            idx_src.append([idx_st for _ in range(e.get_num_points_roi(sim_roi=sim_roi, simul_type=simul_type))])

        return source_term, np.array(idx_src, dtype=np.int32).flatten()
