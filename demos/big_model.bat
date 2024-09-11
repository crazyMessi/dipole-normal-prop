@echo off

REM Navigate to the parent directory and set BASE_PATH
cd ..
for /f "delims=" %%i in ('cd') do set BASE_PATH=%%i
echo %BASE_PATH%

REM Set the PYTHONPATH and MODEL_NAME environment variables
set PYTHONPATH=%BASE_PATH%
set MODEL_NAME=scene0054_00_vh_clean_2


REW print the cmd
echo python -u %BASE_PATH%/orient_large.py ^
--pc %BASE_PATH%/data/%MODEL_NAME%.xyz ^
--export_dir %BASE_PATH%/demos/%MODEL_NAME% ^
--models %BASE_PATH%/pre_trained/hands2.pt ^
%BASE_PATH%/pre_trained/hands.pt ^
%BASE_PATH%/pre_trained\manmade.pt ^
--iters 5 ^
--propagation_iters 4 ^
--number_parts 41 ^
--minimum_points_per_patch 100 ^
--diffuse ^
--weighted_prop ^
--n 50


REM Run the Python script with the necessary arguments
python -u %BASE_PATH%/orient_large.py ^
--pc %BASE_PATH%/data/%MODEL_NAME%.xyz ^
--export_dir %BASE_PATH%/demos/%MODEL_NAME% ^
--models %BASE_PATH%/pre_trained/hands2.pt ^
%BASE_PATH%/pre_trained/hands.pt ^
%BASE_PATH%/pre_trained\manmade.pt ^
--iters 5 ^
--propagation_iters 4 ^
--number_parts 41 ^
--minimum_points_per_patch 100 ^
--diffuse ^
--weighted_prop ^
--n 50
