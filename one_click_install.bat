@echo off
REM Check if the virtual conda environment exists
CALL conda env list | FIND /I "paints_undo" >nul
IF ERRORLEVEL 1 (
    echo Creating virtual conda environment...
    conda create -y -n paints_undo python=3.10
)

REM Activate the virtual conda environment
echo Activating virtual conda environment...
CALL conda activate paints_undo

REM Upgrade pip to the latest version
python.exe -m pip install --upgrade pip --no-cache-dir

REM Download Triton wheel and place it in the main tree folder
echo Downloading Triton==2.1.0 .whl...
curl -L -o triton-2.1.0-cp310-cp310-win_amd64.whl https://huggingface.co/MonsterMMORPG/SECourses/resolve/main/triton-2.1.0-cp310-cp310-win_amd64.whl?download=true

REM Installing torch, dependencies from requirements.txt & triton
echo Installing dependencies...
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
pip install -r requirements.txt --no-cache-dir
pip install triton-2.1.0-cp310-cp310-win_amd64.whl --no-cache-dir

REM Deactivate the virtual environment
echo Deactivating virtual conda environment...
CALL conda deactivate

exit
