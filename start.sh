source ./venv/bin/activate

# 指定配置文件路径
config_file="config.ini"

# 从配置文件中读取信息
host="$(awk -F' = ' '/^host/ {print $2}' <(grep 'host' $config_file))"
port=$(awk -F' = ' '/^port/ {print $2}' <(grep 'port' $config_file))
auth_file_path="$(awk -F' = ' '/^auth_file_path/ {print $2}' <(grep 'auth_file_path' $config_file))"
sd_host="$(awk -F' = ' '/^sd_host/ {print $2}' <(grep 'sd_host' $config_file))"
ollama_host="$(awk -F' = ' '/^ollama_host/ {print $2}' <(grep 'ollama_host' $config_file))"
openai_host="$(awk -F' = ' '/^openai_host/ {print $2}' <(grep 'openai_host' $config_file))"
openai_key="$(awk -F' = ' '/^openai_key/ {print $2}' <(grep 'openai_key' $config_file))"

# 检查port是否已读取
if [ -z "$port" ]; then
  echo "Port not found in configuration file."
  exit 1
fi

echo "Looking for processes using TCP port $port..."

# 使用lsof命令获取特定TCP端口的进程PID
pid=$(lsof -t -i tcp:$port)

# 检查PID是否为空，如果不为空，则输出PID
if [ ! -z "$pid" ]; then
  echo "Found process on port $port with PID: $pid"
  # 如果需要杀死该进程，取消下一行的注释
  kill -9 $pid
else
  echo "No process found on port $port."
fi

sleep 2

nohup python -u app/main-api-webui.py --host $host --port $port --auth_file_path $auth_file_path --sd_host $sd_host --ollama_host $ollama_host --openai_host $openai_host --openai_key $openai_key > nohub.log 2>&1 &

echo "start success!!"
