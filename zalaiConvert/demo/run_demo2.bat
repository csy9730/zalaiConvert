

set path=C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem
rem D:\programdata\Miniconda3\condabin\conda.bat
set PY=G:\rknn_convert\zal_rk130
set path=%PY%;%PY%\scripts;%PY%\DLLs;%PY%\Library\bin;%PY%\bin;%path%;
rem conda activate G:\rknn_convert\zal_rk130
python zalaiConvert\convertWrap.py --config tmp_demo\tmp.rk.json
