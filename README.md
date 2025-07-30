# Desenvolvimento de uma Rede Neural Artificial para Degravação de Depoimentos Judiciais

Este repositório contém o código-fonte e os arquivos desenvolvidos no trabalho de conclusão de curso (TCC) para o curso de Sistemas de Informação, pelo Instituto Federal do Espírito Santo (IFES), Campus Cachoeiro de Itapemirim, focado na criação de uma solução automatizada para degravação de depoimentos judiciais. A proposta consiste no desenvolvimento de uma Rede Neural Artificial, utilizando técnicas combinadas de redes convolucionais (CNN) e redes recorrentes (RNN), para realizar a transcrição automática de áudios de depoimentos judiciais.

O trabalho final pode ser encontrado no repositório do IFES, disponível no [link]([/guides/content/editing-an-existing-page](https://repositorio.ifes.edu.br/xmlui/handle/123456789/5521)).

---
</br>

## 🗂️ Estrutura do Repositório

A seguir, uma explicação detalhada da estrutura de diretórios e arquivos deste projeto:

```grafql
.
├── data
│   ├── input/             # Diretório com os áudios a serem transcritos
│   ├── output/            # Diretório com as transcrições geradas
├── Dataset/               # Diretório para armazenamento do dataset usado para treinamento
├── Models/                # Diretório com os modelos treinados
├── utils/                 # Scripts utilitários (pré-processamento, etc.)
├── configs.py             # Arquivo de configuração do projeto
├── convert_keras_onnx_model.py  # Script para conversão de modelos Keras para ONNX
├── inference_model.py     # Script principal para realizar inferência
├── inference_model_validation.py # Validação e testes do modelo treinado
├── model.py               # Implementação do modelo neural
├── requirements.txt       # Lista de dependências do projeto
├── train.py               # Script para treinamento do modelo
└── README.md              # Este arquivo
```

### Descrição dos Diretórios

- **`data/input/`**: Diretório onde devem ser armazenados os áudios que serão transcritos pelo modelo. O usuário deve garantir que os arquivos estejam neste local antes de rodar os scripts de inferência.
- **`data/output/`**: Após a execução do script de inferência, as transcrições geradas serão salvas nesta pasta.
- **`Dataset/`**: Diretório destinado ao armazenamento do dataset utilizado para o treinamento do modelo. O dataset deve conter áudios e suas respectivas transcrições.
- **`Models/`**: Contém os arquivos dos modelos treinados (arquivos gerados ou carregados, como `.h5` ou `.onnx`).


### Descrição dos Principais Arquivos

- **`configs.py`**: Configurações gerais do projeto, como parâmetros de treinamento e configurações do modelo.
- **`convert_keras_onnx_model.py`**: Script para conversão de modelos do formato Keras (`.h5`) para ONNX, facilitando a integração e uso em outras plataformas.
- **`inference_model.py`**: Realiza a inferência com os modelos treinados, gerando transcrições dos áudios em `data/input/`.
- **`inference_model_validation.py`**: Realiza validações e testes no modelo utilizando datasets previamente definidos.
- **`train.py`**: Script principal para o treinamento do modelo neural. Utiliza os dados presentes no diretório`Dataset/`.
---
</br>

## ⚙️ Requisitos de Instalação

Antes de executar o projeto, instale as dependências listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

Certifique-se de ter as seguintes ferramentas e bibliotecas instaladas:
- Python 3.9.20
- TensorFlow 2.10.0
---
</br>

## 🚀 Uso do Projeto

### 1. Inferência (Transcrição de Áudios)
1. Coloque os arquivos de áudio a serem transcritos no diretório `data/input/`.
2. Execute o script de inferência:
   ```bash
   python inference_model.py
   ```
3. Os arquivos transcritos serão salvos no caminho `data/input/`.

### 2. Treinamento
1. Adicione o dataset (áudios e transcrições) no diretório `Dataset/`.
2. Configure os parâmetros de treinamento no arquivo configs.py, se necessário.
3. Execute o script de treinamento:
    ```bash
   python train.py
   ```
   Obs.: Nesta etapa é altamente recomendado o uso de GPU para acelerar o processo.
4. Após o treinamento, os modelos gerados serão salvos em `Models/`

---
</br>

## 👩‍💻 Autora
  <img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/73599857?v=4" width="100px;" alt=""/>


Feito com ❤️ por Gabriely Machado Carrari </br>
Entre em contato! 👋🏽



[![Static Badge](https://img.shields.io/badge/Gabriely%20Carrari-%230A66C2?logo=linkedIn&link=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fgabriely-carrari%2F)](https://www.linkedin.com/in/gabriely-carrari/)
[![Static Badge](https://img.shields.io/badge/gabrielycarrari%40gmail.com-%23EA4335?logo=gmail&logoColor=white&link=mailto%3Agabrielycarrari%40gmail.com)](mailto:gabrielycarrari@gmail.com)

---
</br>

## 📃 Licença
Esse repositório está licenciado pela .... . Para mais informações detalhadas, leia o arquivo LICENSE contido nesse repositório.
