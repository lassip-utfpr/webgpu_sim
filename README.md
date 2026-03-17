# webgpu_sim

Repositório de simulação numérica de propagação de ondas (acústicas/elásticas/viscoelásticas) acelerada por GPU com **WebGPU** em Python.

## O que este projeto faz

- Executa simulações em 1D, 2D e 3D usando kernels WGSL (`shader_*.wgsl`).
- Implementa condições de contorno absorventes CPML.
- Permite configuração por arquivos JSON (`config*.json` e variantes).
- Inclui exemplos simples para validar a pilha WebGPU (`hello_compute.py`, `simple_compute*.py`).

## Estrutura principal

- `sim_1D_isotropic_cpml.py`  
  Simulação 1D isotrópica com CPML.
- `full_sim_webgpu_2D_isotropic_cpml.py` / `full_sim_webgpu_2D_viscoelastic_cpml.py`  
  Simulações 2D principais.
- `full_sim_webgpu_3D_isotropic_cpml.py` / `full_sim_webgpu_3D_viscoelastic_cpml.py`  
  Simulações 3D principais.
- `simul_utils.py`  
  Classes utilitárias de ROI, sondas e apoio à simulação.
- `shader_*.wgsl`  
  Shaders de compute executados na GPU.
- `results/`  
  Saídas/resultados de execuções.

## Dependências

As dependências Python estão em `requirements.txt`:

- wgpu
- numpy
- matplotlib
- scipy
- scikit-learn
- pyqtgraph
- PyQt6

## Execução rápida

1. Crie e ative um ambiente virtual Python.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Teste básico de WebGPU:

```bash
python hello_compute.py
```

Saída esperada: uma lista como `[0, 1, 2, 3]`, indicando cópia de buffer via GPU funcionando.

## Observações

- O projeto depende de suporte a WebGPU no ambiente/driver de GPU.
- Scripts de simulação 2D/3D são mais pesados e podem exigir bastante memória.
- Parte dos scripts exibe visualização usando PyQtGraph/PyQt6.
