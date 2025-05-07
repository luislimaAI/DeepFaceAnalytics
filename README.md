# Face Counter & Emotion Analyzer

## Descrição

O Face Counter & Emotion Analyzer é uma aplicação de visão computacional que detecta, reconhece e analisa faces em tempo real através da webcam. O sistema é capaz de:

- Detectar faces em tempo real usando a webcam
- Reconhecer pessoas já identificadas anteriormente
- Analisar emoções/sentimentos detectados (feliz, triste, neutro, etc.)
- Estimar a idade das pessoas detectadas
- Gerar estatísticas e gráficos de emoções predominantes
- Salvar resultados em formato JSON para análise posterior

## Estrutura do Projeto

```
deepface_test/
  ├── face_counter_emotions.py   # Script principal
  ├── requirements.txt           # Lista de dependências
  ├── setup.sh                   # Script de instalação para macOS/Linux
  ├── setup.bat                  # Script de instalação para Windows
  ├── images/                    # Pasta para armazenar imagens
  │   └── known_faces/           # Banco de faces conhecidas
  ├── logs/                      # Registros de execução
  └── output/                    # Arquivos JSON com resultados
```

## Requisitos

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib
- DeepFace
- TensorFlow
- tf-keras
- (outras dependências estão listadas no arquivo requirements.txt)

## Instalação

### Instalação Rápida (Recomendada)

#### No Windows:
1. Dê um duplo clique no arquivo `setup.bat`
2. Aguarde a conclusão da instalação

#### No macOS/Linux:
1. Abra o terminal na pasta do projeto
2. Execute o comando:
   ```bash
   chmod +x setup.sh && ./setup.sh
   ```
3. Aguarde a conclusão da instalação

O script de instalação irá:
- Criar um ambiente virtual
- Instalar todas as dependências necessárias
- Configurar tudo para o funcionamento adequado do sistema

### Configurando um Ambiente Virtual Manualmente

É altamente recomendável usar um ambiente virtual para evitar conflitos de dependências entre projetos:

1. Instale o virtualenv se ainda não estiver instalado:
   ```bash
   pip install virtualenv
   ```

2. Crie um ambiente virtual na pasta do projeto:
   ```bash
   # No Windows
   python -m venv venv
   
   # No macOS/Linux
   python3 -m venv venv
   ```

3. Ative o ambiente virtual:
   ```bash
   # No Windows
   venv\Scripts\activate
   
   # No macOS/Linux
   source venv/bin/activate
   ```

4. Depois de ativado, você verá o nome do ambiente (venv) no início da linha de comando

### Instalando as Dependências Manualmente

Com o ambiente virtual ativado:

1. Clone este repositório:
   ```bash
   git clone [url-do-repositorio]
   ```

2. Instale todas as dependências de uma vez:
   ```bash
   pip install -r requirements.txt
   ```
   
   **Nota para TensorFlow 2.19+**: Se estiver usando TensorFlow 2.19 ou superior, você precisará instalar o pacote tf-keras que já está incluído no requirements.txt:
   ```bash
   pip install tf-keras
   ```
   
3. Para sair do ambiente virtual quando terminar:
   ```bash
   deactivate
   ```

## Uso

Com o ambiente virtual ativado, execute o script principal:

```bash
python face_counter_emotions.py
```

### Menu de Opções

1. **Detectar e contar pessoas em tempo real (webcam)**
   - Inicia a webcam e detecta faces em tempo real
   - Mostra informações sobre emoções e idades na tela
   - Use 'q' para sair, 's' para salvar um frame, 'e' para ver estatísticas

2. **Ver estatísticas de sentimentos**
   - Exibe estatísticas detalhadas sobre as emoções detectadas
   - Mostra a distribuição percentual de emoções por pessoa
   - Cria e exibe gráficos de sentimentos e idades médias

3. **Ver contagem de pessoas identificadas**
   - Lista todas as pessoas detectadas
   - Mostra o sentimento predominante de cada pessoa
   - Apresenta estatísticas de idade

4. **Salvar resultados em JSON**
   - Gera um arquivo JSON na pasta 'output' com todas as informações
   - Inclui timestamp, distribuição de emoções e idades médias

5. **Sair**
   - Encerra a aplicação

## Formato do JSON de Saída

Os resultados são salvos em formato JSON na pasta `output/` com a seguinte estrutura:

```json
{
    "timestamp": "2025-05-07 12:00:00",
    "date": "2025-05-07",
    "time": "12:00:00",
    "session_info": {
        "total_detected_faces": 50,
        "unique_people": 2,
        "duration_seconds": 120.5
    },
    "people": [
        {
            "id": "face_20250507_120000_1",
            "name": "Pessoa 1",
            "detection_count": 30,
            "first_seen": "2025-05-07 11:58:00",
            "last_seen": "2025-05-07 12:00:00",
            "emotions": {
                "predominant": "happy",
                "distribution": {
                    "happy": 60.0,
                    "neutral": 30.0,
                    "surprise": 10.0
                }
            },
            "age": {
                "average": 28.5,
                "samples": 30
            }
        }
    ]
}
```

## Solução de Problemas Comuns

### Erros com TensorFlow/DeepFace
Se encontrar erros relacionados ao TensorFlow ou DeepFace, tente:
```bash
pip install tf-keras
```

### Problemas com Acesso à Webcam
Verifique se sua webcam está funcionando e não está sendo usada por outro aplicativo.

### Erro "No module named..."
Certifique-se de que o ambiente virtual está ativado e todas as dependências foram instaladas:
```bash
pip install -r requirements.txt
```

## Funcionalidades Adicionais

- **Janela deslizante para emoções**: O sistema usa uma janela deslizante para determinar o sentimento predominante atual, tornando a detecção mais precisa ao longo do tempo
- **Persistência de dados**: As faces conhecidas são salvas para reconhecimento em execuções futuras
- **Gráficos visuais**: Visualização clara da distribuição de emoções e idades médias
- **Exportação em JSON**: Dados estruturados para análise posterior ou integração com outros sistemas