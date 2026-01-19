#!/usr/bin/env bash
set -euo pipefail

# 一键运行：自动建立 SSH 隧道 + 启动推理桥接脚本 examples/run_openpi_inference.py
# 运行方式：
#   bash examples/run_openpi_inference.sh
#
# 你只需要修改下面的“参数区”，每个参数后面都有中文注释。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

###############################################################################
# 参数区（按需修改）
###############################################################################

PYTHON_BIN="${PYTHON_BIN:-python}" # Python 解释器（例如 conda 环境里的 python）；也可用环境变量覆盖

FRANKA_URL="${FRANKA_URL:-http://127.0.0.2:5000/}" # 机器人端 franka_server.py 的 Flask 地址（例如 http://127.0.0.1:5000/；也可用环境变量覆盖）

OPENPI_REMOTE_HOST="${OPENPI_REMOTE_HOST:-172.24.178.135}" # 模型端 OpenPI 服务所在机器/节点的可达 IP/域名（从跳板机侧能访问；也可用环境变量覆盖）
OPENPI_REMOTE_PORT="${OPENPI_REMOTE_PORT:-8000}"           # 模型端 OpenPI websocket 端口（你的日志显示监听 0.0.0.0:8000；也可用环境变量覆盖）

OPENPI_LOCAL_PORT="${OPENPI_LOCAL_PORT:-9000}"             # 本机用于端口转发的端口（转发后本机用 127.0.0.1:PORT 访问；也可用环境变量覆盖）

OPENPI_SSH_HOST="${OPENPI_SSH_HOST:-ssh.platform-sz.jingneng-inner.ac.cn}" # 跳板机 host（不要求本机 DNS 可解析，走 ProxyCommand；也可用环境变量覆盖）
OPENPI_SSH_USER="${OPENPI_SSH_USER:-experiment3.zhaomingxuan.research-prod_flageval.cn-beijing-shangzhuang.ws}" # 跳板机登录用户名（也可用环境变量覆盖）
OPENPI_SSH_PORT="${OPENPI_SSH_PORT:-2222}"                                 # 跳板机 SSH 端口（也可用环境变量覆盖）
OPENPI_PROXY_COMMAND="${OPENPI_PROXY_COMMAND:-ssh -p 10025 work@120.92.17.239 -W %h:%p}" # 你的 ProxyCommand（二段跳；也可用环境变量覆盖）

HZ="${HZ:-10}"                      # 控制循环频率（Hz；每个控制步的频率；也可用环境变量覆盖）
EXEC_HORIZON="${EXEC_HORIZON:-50}"  # 每次拿到一次模型推理结果后，连续执行多少个控制步（例如 50；也可用环境变量覆盖）
MAX_STEPS="${MAX_STEPS:-0}"         # 最大步数（0 表示一直运行；也可用环境变量覆盖）
GRIPPER_OPEN_THRESHOLD="${GRIPPER_OPEN_THRESHOLD:-0.7}" # 模型动作中 gripper 阈值：>=0.7 开夹爪，<0.7 合夹爪（也可用环境变量覆盖）

PROMPT="${PROMPT:-Pick up the building block and put it into the basket}"               # 可选：prompt（留空表示不传；也可用环境变量覆盖）
BGR_TO_RGB="${BGR_TO_RGB:-0}"      # 是否把相机 BGR 转 RGB：0=不转；1=转（也可用环境变量覆盖）
RESIZE_HW="${RESIZE_HW:-}"         # 可选：图像 resize 到 "H W"（例如 "224 224"）；留空表示不 resize（也可用环境变量覆盖）

SAVE_OBS="${SAVE_OBS:-1}"          # 是否实时保存观测：1=保存；0=不保存（也可用环境变量覆盖）
SAVE_OBS_DIR="${SAVE_OBS_DIR:-saved_obs}"   # 观测保存目录（也可用环境变量覆盖）
SAVE_OBS_EVERY="${SAVE_OBS_EVERY:-1}"       # 每隔多少控制步保存一次：1=每步都保存；10=每 10 步保存一次（也可用环境变量覆盖）

FRONT_SERIAL="${FRONT_SERIAL:-}"   # 可选：前视相机 serial（留空要求只接 2 个 RealSense，脚本自动选；也可用环境变量覆盖）
WRIST_SERIAL="${WRIST_SERIAL:-}"   # 可选：腕部相机 serial（同上）
CAMERA_ORDER="${CAMERA_ORDER:-front_first}" # 自动选相机时的枚举顺序：front_first=第1个当前视、第2个当腕部；wrist_first 反过来（也可用环境变量覆盖）

TUNNEL_WAIT_S="${TUNNEL_WAIT_S:-1.0}" # SSH 隧道启动后等待秒数（给端口转发一点启动时间；也可用环境变量覆盖）

###############################################################################
# 实现区（一般不需要改）
###############################################################################

OPENPI_WS="ws://127.0.0.1:${OPENPI_LOCAL_PORT}" # 推理脚本连接的 websocket 地址（本地端口转发）

CONTROL_PATH="/tmp/openpi_tunnel_${OPENPI_LOCAL_PORT}_$$.sock" # SSH 控制 socket（用于优雅关闭隧道）

cleanup() {
  # 优先通过 ControlMaster 关闭隧道（最稳）；失败则忽略
  ssh -S "${CONTROL_PATH}" -O exit \
    -p "${OPENPI_SSH_PORT}" \
    -o "ProxyCommand=${OPENPI_PROXY_COMMAND}" \
    "${OPENPI_SSH_USER}@${OPENPI_SSH_HOST}" >/dev/null 2>&1 || true
  rm -f "${CONTROL_PATH}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[run] FRANKA_URL=${FRANKA_URL}"
echo "[run] OPENPI_REMOTE=${OPENPI_REMOTE_HOST}:${OPENPI_REMOTE_PORT}"
echo "[run] OPENPI_LOCAL=${OPENPI_WS}"
echo "[run] SSH=${OPENPI_SSH_USER}@${OPENPI_SSH_HOST}:${OPENPI_SSH_PORT}"

echo "[run] starting ssh tunnel...（如提示 Password，请输入；成功后 ssh 会自动后台运行）"
ssh -fN \
  -p "${OPENPI_SSH_PORT}" \
  -o "ProxyCommand=${OPENPI_PROXY_COMMAND}" \
  -o "ControlMaster=yes" \
  -o "ControlPath=${CONTROL_PATH}" \
  -o "ControlPersist=yes" \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L "${OPENPI_LOCAL_PORT}:${OPENPI_REMOTE_HOST}:${OPENPI_REMOTE_PORT}" \
  "${OPENPI_SSH_USER}@${OPENPI_SSH_HOST}"

sleep "${TUNNEL_WAIT_S}"

if command -v curl >/dev/null 2>&1; then
  echo "[run] checking openpi healthz via tunnel..."
  ok=0
  for _ in $(seq 1 20); do
    if curl -sSf --max-time 1 "http://127.0.0.1:${OPENPI_LOCAL_PORT}/healthz" >/dev/null; then
      ok=1
      break
    fi
    sleep 0.3
  done
  if [[ "${ok}" != "1" ]]; then
    echo "[error] healthz check failed. local_port=${OPENPI_LOCAL_PORT} remote=${OPENPI_REMOTE_HOST}:${OPENPI_REMOTE_PORT}" >&2
    echo "[hint] 可能原因：1) SSH 密码/权限不对 2) 跳板机无法访问模型端 ${OPENPI_REMOTE_HOST}:${OPENPI_REMOTE_PORT} 3) 模型端服务未起来/端口不对" >&2
    exit 1
  fi
fi

if command -v curl >/dev/null 2>&1; then
  echo "[run] checking franka server..."
  if ! curl -sSf --max-time 2 -X POST "${FRANKA_URL%/}/status" >/dev/null; then
    echo "[error] cannot reach franka server at ${FRANKA_URL%/} (POST /status failed)" >&2
    echo "[hint] 请确认机器人端 franka_server.py 已启动，并把 FRANKA_URL 设成它实际监听的地址，例如：" >&2
    echo "       FRANKA_URL='http://127.0.0.1:5000/' bash examples/run_openpi_inference.sh" >&2
    exit 1
  fi
fi

args=(
  --franka_url "${FRANKA_URL}"
  --openpi_ws "${OPENPI_WS}"
  --hz "${HZ}"
  --exec_horizon "${EXEC_HORIZON}"
  --max_steps "${MAX_STEPS}"
  --gripper_open_threshold "${GRIPPER_OPEN_THRESHOLD}"
  --save_obs_dir "${SAVE_OBS_DIR}"
  --save_obs_every "${SAVE_OBS_EVERY}"
)

if [[ "${SAVE_OBS}" == "0" ]]; then args+=(--no-save_obs); fi
if [[ -n "${FRONT_SERIAL}" ]]; then args+=(--front_serial "${FRONT_SERIAL}"); fi
if [[ -n "${WRIST_SERIAL}" ]]; then args+=(--wrist_serial "${WRIST_SERIAL}"); fi
if [[ -n "${CAMERA_ORDER}" ]]; then args+=(--camera_order "${CAMERA_ORDER}"); fi
if [[ -n "${PROMPT}" ]]; then args+=(--prompt "${PROMPT}"); fi
if [[ "${BGR_TO_RGB}" == "1" ]]; then args+=(--bgr_to_rgb); fi
if [[ -n "${RESIZE_HW}" ]]; then
  read -r RESIZE_H RESIZE_W <<<"${RESIZE_HW}"
  args+=(--resize_hw "${RESIZE_H}" "${RESIZE_W}")
fi

echo "[run] starting inference bridge..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_openpi_inference.py" "${args[@]}" "$@"
