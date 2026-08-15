#!/usr/bin/env bash
set -euo pipefail

INSTANCE=${INSTANCE:-llm-week2-kind}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAB_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
RESULT_DIR=$LAB_DIR/results
mkdir -p "$RESULT_DIR"

lxc exec "$INSTANCE" -- bash -s <<'INNER'
set -euo pipefail
service=week2-rayservice-serve-svc
curl_pod=week2-rayservice-curl

kubectl delete pod "$curl_pod" --ignore-not-found --wait=true >/dev/null
kubectl run "$curl_pod" --restart=Never --image=curlimages/curl:8.16.0 \
  --command -- sleep 900 >/dev/null
kubectl wait --for=condition=Ready "pod/$curl_pod" --timeout=120s >/dev/null

probe_until_stopped() {
  local output=$1 stop_file=$2 interval=$3
  printf 'elapsed_seconds\thttp_ok\tresponse\n' > "$output"
  local start now elapsed response ok
  start=$(date +%s)
  for _ in $(seq 1 240); do
    response=$(kubectl exec "$curl_pod" -- curl -fsS --max-time 2 \
      -X POST -H 'Content-Type: application/json' --data '["MANGO", 2]' \
      "http://${service}:8000/fruit/" 2>/dev/null || true)
    ok=0
    [[ $response == 6 ]] && ok=1
    now=$(date +%s)
    elapsed=$((now - start))
    printf '%s\t%s\t%s\n' "$elapsed" "$ok" "${response//$'\n'/ }" >> "$output"
    [[ -e $stop_file ]] && break
    sleep "$interval"
  done
}

rm -f /tmp/week2-upgrade.stop /tmp/week2-upgrade-probes.tsv \
  /tmp/week2-upgrade-transitions.tsv
old_cluster=$(kubectl get raycluster -o jsonpath='{.items[0].metadata.name}')
probe_until_stopped /tmp/week2-upgrade-probes.tsv /tmp/week2-upgrade.stop 1 &
probe_pid=$!
start=$(date +%s)
printf 'elapsed_seconds\tservice_status\trayclusters\n' > /tmp/week2-upgrade-transitions.tsv

revision="kind-$(date +%s)"
kubectl patch rayservice week2-rayservice --type=json -p="[
  {\"op\":\"replace\",\"path\":\"/spec/rayClusterConfig/headGroupSpec/template/spec/containers/0/env/0/value\",\"value\":\"${revision}\"},
  {\"op\":\"replace\",\"path\":\"/spec/rayClusterConfig/workerGroupSpecs/0/template/spec/containers/0/env/0/value\",\"value\":\"${revision}\"}
]" >/dev/null

new_cluster=''
for _ in $(seq 1 150); do
  now=$(date +%s)
  elapsed=$((now - start))
  status=$(kubectl get rayservice week2-rayservice \
    -o jsonpath='{.status.serviceStatus}' 2>/dev/null || true)
  clusters=$(kubectl get raycluster -o jsonpath='{range .items[*]}{.metadata.name}{" "}{end}' 2>/dev/null || true)
  printf '%s\t%s\t%s\n' "$elapsed" "$status" "$clusters" >> /tmp/week2-upgrade-transitions.tsv
  for cluster in $clusters; do
    [[ $cluster != "$old_cluster" ]] && new_cluster=$cluster
  done
  if [[ -n $new_cluster && $status == Running ]]; then
    ready=$(kubectl get raycluster "$new_cluster" -o jsonpath='{.status.state}' 2>/dev/null || true)
    [[ $ready == ready ]] && break
  fi
  sleep 2
done
[[ -n $new_cluster && ${ready:-} == ready && ${status:-} == Running ]]
upgrade_seconds=$(( $(date +%s) - start ))
sleep 10
touch /tmp/week2-upgrade.stop
wait "$probe_pid"
upgrade_success=$(awk -F '\t' 'NR>1 && $2==1 {n++} END {print n+0}' /tmp/week2-upgrade-probes.tsv)
upgrade_failure=$(awk -F '\t' 'NR>1 && $2==0 {n++} END {print n+0}' /tmp/week2-upgrade-probes.tsv)

# KubeRay가 이전 cluster를 정리한 뒤 active cluster의 worker만 대상으로 삼는다.
for _ in $(seq 1 120); do
  clusters=$(kubectl get raycluster -o jsonpath='{range .items[*]}{.metadata.name}{" "}{end}')
  [[ $clusters == "${new_cluster} " ]] && break
  sleep 2
done
[[ $clusters == "${new_cluster} " ]]

rm -f /tmp/week2-recovery.stop /tmp/week2-recovery-probes.tsv
probe_until_stopped /tmp/week2-recovery-probes.tsv /tmp/week2-recovery.stop 1 &
probe_pid=$!
old_worker=$(kubectl get pod -l "ray.io/cluster=${new_cluster},ray.io/node-type=worker" \
  -o jsonpath='{.items[0].metadata.name}')
start=$(date +%s)
kubectl delete pod "$old_worker" --wait=false >/dev/null
new_worker=''
for _ in $(seq 1 150); do
  new_worker=$(kubectl get pod -l "ray.io/cluster=${new_cluster},ray.io/node-type=worker" \
    -o jsonpath="{range .items[?(@.metadata.name!='${old_worker}')]}{.metadata.name}{end}" \
    2>/dev/null || true)
  if [[ -n $new_worker ]]; then
    ready=$(kubectl get pod "$new_worker" \
      -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null || true)
    status=$(kubectl get rayservice week2-rayservice \
      -o jsonpath='{.status.serviceStatus}' 2>/dev/null || true)
    [[ $ready == true && $status == Running ]] && break
  fi
  sleep 2
done
[[ -n $new_worker && ${ready:-} == true && ${status:-} == Running ]]
recovery_seconds=$(( $(date +%s) - start ))
sleep 10
touch /tmp/week2-recovery.stop
wait "$probe_pid"
recovery_success=$(awk -F '\t' 'NR>1 && $2==1 {n++} END {print n+0}' /tmp/week2-recovery-probes.tsv)
recovery_failure=$(awk -F '\t' 'NR>1 && $2==0 {n++} END {print n+0}' /tmp/week2-recovery-probes.tsv)

cat > /tmp/week2-cpu-lifecycle.env <<EOF
OLD_CLUSTER=$old_cluster
NEW_CLUSTER=$new_cluster
UPGRADE_SECONDS=$upgrade_seconds
UPGRADE_SUCCESS=$upgrade_success
UPGRADE_FAILURE=$upgrade_failure
OLD_WORKER=$old_worker
NEW_WORKER=$new_worker
RECOVERY_SECONDS=$recovery_seconds
RECOVERY_SUCCESS=$recovery_success
RECOVERY_FAILURE=$recovery_failure
EOF
cat /tmp/week2-cpu-lifecycle.env
INNER

for file in week2-upgrade-probes.tsv week2-upgrade-transitions.tsv \
  week2-recovery-probes.tsv week2-cpu-lifecycle.env; do
  lxc file pull "$INSTANCE/tmp/$file" "$RESULT_DIR/$file"
done
