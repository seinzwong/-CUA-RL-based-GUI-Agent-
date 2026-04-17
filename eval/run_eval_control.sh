# terminal 1 (Feishu CUA default)
save_path=eval/feishu_cua_eval/
logs_path=${save_path}logs

mkdir -p ${logs_path}

# launch controller
python -u -m fastchat.serve.controller --host 0.0.0.0