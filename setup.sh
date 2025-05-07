#!/bin/bash

# Script de instalação para macOS e Linux
echo "=== Iniciando instalação do ambiente para o Face Counter & Emotion Analyzer ==="

# Verificar se o Python está instalado
command -v python3 >/dev/null 2>&1 || { echo "Python 3 não está instalado. Por favor, instale-o primeiro."; exit 1; }

echo "Criando ambiente virtual..."
python3 -m venv venv

echo "Ativando ambiente virtual..."
source venv/bin/activate

echo "Atualizando pip..."
pip install --upgrade pip

echo "Instalando todas as dependências..."
pip install -r requirements.txt

echo "=== Instalação concluída! ==="
echo "Para ativar o ambiente virtual, execute: source venv/bin/activate"
echo "Para executar o programa, use: python face_counter_emotions.py"