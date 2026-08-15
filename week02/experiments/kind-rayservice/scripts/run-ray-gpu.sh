#!/usr/bin/env bash
set -euo pipefail

INSTANCE=${INSTANCE:-llm-week2-kind}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAB_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
MANIFEST=$LAB_DIR/manifests/rayservice-gpu.yaml

old_cluster=$(lxc exec "$INSTANCE" -- kubectl get rayservice week2-rayservice \
  -o jsonpath='{.status.activeServiceStatus.rayClusterName}' 2>/dev/null || true)
old_desired_gpu=$(lxc exec "$INSTANCE" -- kubectl get rayservice week2-rayservice \
  -o jsonpath='{.status.activeServiceStatus.rayClusterStatus.desiredGPU}' 2>/dev/null || true)
lxc exec "$INSTANCE" -- rm -f /tmp/rayservice-gpu.yaml
lxc file push "$MANIFEST" "$INSTANCE/tmp/rayservice-gpu.yaml"
lxc exec "$INSTANCE" -- kubectl apply -f /tmp/rayservice-gpu.yaml

for _ in $(seq 1 180); do
  status=$(lxc exec "$INSTANCE" -- kubectl get rayservice week2-rayservice \
    -o jsonpath='{.status.serviceStatus}' 2>/dev/null || true)
  active_cluster=$(lxc exec "$INSTANCE" -- kubectl get rayservice week2-rayservice \
    -o jsonpath='{.status.activeServiceStatus.rayClusterName}' 2>/dev/null || true)
  desired_gpu=$(lxc exec "$INSTANCE" -- kubectl get rayservice week2-rayservice \
    -o jsonpath='{.status.activeServiceStatus.rayClusterStatus.desiredGPU}' 2>/dev/null || true)
  [[ $status == Running && -n $active_cluster && $desired_gpu == 1 && \
     ( $old_desired_gpu == 1 || $active_cluster != "$old_cluster" ) ]] && break
  sleep 5
done
[[ ${status:-} == Running && -n ${active_cluster:-} && ${desired_gpu:-} == 1 && \
   ( $old_desired_gpu == 1 || ${active_cluster:-} != "$old_cluster" ) ]]

lxc exec "$INSTANCE" -- bash -s -- "$active_cluster" <<'INNER'
set -euo pipefail
active_cluster=$1
head=$(kubectl get pod -l "ray.io/cluster=${active_cluster},ray.io/node-type=head" \
  -o jsonpath='{.items[0].metadata.name}')
kubectl exec -i "$head" -c ray-head -- python - <<'PY'
import json
import os
from pathlib import Path
import ray

ray.init(address="auto")

@ray.remote(num_cpus=0.1, num_gpus=1)
def inspect_gpu():
    return {
        "ray_gpu_ids": ray.get_runtime_context().get_accelerator_ids().get("GPU", []),
        "dev_kfd": Path("/dev/kfd").exists(),
        "dri_devices": sorted(str(p) for p in Path("/dev/dri").glob("*")),
        "visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
    }

result = ray.get(inspect_gpu.remote())
print(json.dumps(result, sort_keys=True))
assert result["ray_gpu_ids"] == ["0"] or result["ray_gpu_ids"] == [0]
assert result["dev_kfd"]
PY
INNER
