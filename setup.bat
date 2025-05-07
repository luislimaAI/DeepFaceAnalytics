@echo off
echo === Iniciando instalacao do ambiente para o Face Counter ^& Emotion Analyzer ===

REM Verificar se o Python esta instalado
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python nao esta instalado. Por favor, instale-o primeiro.
    exit /b 1
)

echo Criando ambiente virtual...
python -m venv venv

echo Ativando ambiente virtual...
call venv\Scripts\activate.bat

echo Atualizando pip...
python -m pip install --upgrade pip

echo Instalando todas as dependencias...
pip install -r requirements.txt

echo === Instalacao concluida! ===
echo Para ativar o ambiente virtual, execute: venv\Scripts\activate.bat
echo Para executar o programa, use: python face_counter_emotions.py
pause