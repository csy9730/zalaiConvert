# rknn.bin.visualization

## arch

可视化工具visualization，可以执行模型转化功能。

该工具是CS架构。通过引导脚本启动窗口程序和后台flask服务。

引导进程：`rknn\bin\visualization.py` 脚本
窗口进程：window `rknn\visualization\front_end_win\rknn.exe`
服务进程：后台flask服务,该服务可以启动模型转换子进程。


### window

rknn.exe 窗口程序位于：`Lib\site-packages\rknn\visualization\front_end_win\rknn.exe`

该程序是基于electron框架构建，使用html5技术实现桌面界面程序。

### visualization
`Lib\site-packages\rknn\bin\visualization.py`，该脚本是引导程序，可以启动`rknn.exe`和`flask`服务。

``` python
def start_run_server(server_flag_file, host, PORT, server_log, MAX_WINDOW):
    # 开启最小server端
    s_p = Process(target=RKNNtoolkit_server, args=(host, PORT, server_log, MAX_WINDOW), name='RKNNtoolkit_server')
    s_p.start()

def open_window(host, port):
    os.chdir(os.path.dirname(__file__))
    print('*********************** open window ***********************')
    try:
        if platform.system() == 'Windows':
            os.system("..\\visualization\\front_end_win\\RKNN.exe 0 http://" + str(host) + ':' + str(port))
    except Exception as err:
        print(err)


def main():
    PORT = 7000
    MAX_WINDOW = 1
    host = '127.0.0.1'
    server_log = 'RKNN_toolkit.log'

    server_flag_file = 'rknn_toolkit_server_flag.json'

    if os.path.exists(server_flag_file):
        # server_flag_file存在：
        #   1.已开启server端；
        #   2.没有开启server端, 程序异常退出, server_flag_file没有删除；
        #   1. 已开启server端；
        if server_flag['BIN_PID'] in psutil.pids():
            open_window(host, PORT)
            # w_t = Thread(target=open_window, args=(host, PORT))
            # w_t.start()
        else:
            # 选择一个没有被占用的端口
            while (net_is_used(PORT, ip=host)):
                PORT += 1
            w_t = Thread(target=open_window, args=(host, PORT))
            w_t.start()

            print('server_flag_file exist,run server first time')
            server_pid = start_run_server(server_flag_file, host, PORT, server_log, MAX_WINDOW)
            w_t.join()
            kill_process(server_pid, server_flag_file)
    else:
        # server_flag_file不存在:
        #   1.第一次运行server端；
        #   2.非第一次运行server端，但server端正常关闭

        # 选择一个没有被占用的端口
        while(net_is_used(PORT, ip=host)):
            PORT += 1

        w_t = Thread(target=open_window, args=(host, PORT))
        w_t.start()

        print('server_flag_file doesn\'t exist, run server first time')
        server_pid = start_run_server(server_flag_file, host, PORT, server_log, MAX_WINDOW)
        w_t.join()
        kill_process(server_pid, server_flag_file)
		
```


### flask

~\Lib\site-packages\rknn\visualization\server\flask_rknn_tookit.py
``` python
			
def init_flask(app, log_dict, start_convert_q, convert_to_RKNN_para, sys_log_file,
               has_RKNN_model_path, per_window_process):
	from .rknn_func import start_normal_convert, start_quantization_convert, \
        start_has_RKNNmodel, start_get_deviceId, start_has_RKNNmodel_get_model_json		   
    # 加载模型
    @app.route('/api/convert_to_RKNNmodel/convert/<window>', methods=['post'])
    def convert(window): 
            p = Process(target=start_normal_convert,
                        args=(parameters, start_convert_q[window], log_dict[window], my_log_file, tmpfile_dir))
            p.start()
            per_window_process[str(window)].append(p.pid)
			
    # 获取日志
    @app.route('/api/get_log/<window>', methods=['post'])
    def get_log(window):
	
	
def RKNNtoolkit_server(host='127.0.0.1', port=5000, sys_log_file='RKNN_toolkit.log', MAX_WINDOW=5):
    app = Flask(__name__)


    # 打开一个新的窗口后，进行窗口初始化
    @app.route('/api/open_window_init/<window>', methods=['get', 'post'])
    def open_window_init(window):
	
    # 关闭当前窗口
    @app.route('/api/close_window/<window>', methods=['get', 'post'])
    def close_window(window):
	
    #server端初始化
    @app.route('/api/init', methods=['get', 'post'])
    def init():
        init_flask(app, log_dict, start_convert_q, convert_to_RKNN_para, sys_log_file,
                       has_RKNN_model_path, per_window_process)
        return 'OK'


    @app.route('/', methods=['get'])
    def test():
        return 'RKNNtoolkit'


    app.run(host=host, port=port)
```

