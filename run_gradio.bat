@echo off
REM Get the directory of the script
SET script_dir=%~dp0

REM Navigate to the script directory
cd /d %script_dir%

REM Activate the virtual conda environment
echo Activating virtual conda environment...
CALL conda activate paints_undo

REM Run the Gradio app script
echo Running Gradio app...
python gradio_app.py

REM Deactivate the virtual environment after the script finishes
echo Deactivating virtual conda environment...
CALL conda deactivate

pause
exit