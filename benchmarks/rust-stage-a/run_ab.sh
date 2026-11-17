#!/usr/bin/env bash
# A/B session for the Rust engine work: the pre-change commit ("old") and the working tree ("new") run
# back to back, one timed job at a time, on the same machine. Usage:
#   OLD=<dir holding old build's target/release/examples> NEW=<same for new> NEWEB=<new engine_ab dir> bash run_ab.sh
# OLD  = a build of the commit before the change (git archive HEAD~ of isongraph-vector-rs, CARGO_TARGET_DIR=...)
# NEW  = cargo build --release --examples in isongraph-vector-rs
# NEWEB= engine_ab built with RUSTFLAGS="--cfg has_batch" (adds the batch experiments)
set -u
DATA=/d/Goldmine/ISON/bench_data/prep
OUT="$(cd "$(dirname "$0")" && pwd)/raw"
wait_quiet() {
  for i in $(seq 1 30); do
    load=$(powershell -NoProfile -Command "[int](Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 2 -MaxSamples 2 | % { \$_.CounterSamples.CookedValue } | Measure-Object -Average).Average")
    if [ "${load:-100}" -lt 30 ]; then echo "  [cpu load before: ${load}%]"; return; fi
    sleep 10
  done
  echo "  [cpu load stayed high: ${load}%]"
}
cb() { # which exe, label, dataset, n, out-suffix
  wait_quiet; echo "== competitive_bench $2 $3 $4"
  "$1/competitive_bench.exe" --ds-dir $DATA/$3 --ds $3 --n $4 --exp core,mutation --queries 2000 --budget 3 --label $2 --out "$OUT/cb_$2_$3_$4$5.json" > "$OUT/cb_$2_$3_$4$5.txt" 2>&1; tail -3 "$OUT/cb_$2_$3_$4$5.txt"
}
eb() { # exe, label, n, exps, suffix, extra args
  wait_quiet; echo "== engine_ab $2 n=$3 $4"
  "$1/engine_ab.exe" --ds-dir $DATA/text384 --n $3 --exp $4 --queries 2000 --label $2 --out "$OUT/eb_$2_$3_$4$5.json" ${6:-} > "$OUT/eb_$2_$3_$4$5.txt" 2>&1; tail -20 "$OUT/eb_$2_$3_$4$5.txt"
}
step=${1:-all}
if [ "$step" = all ] || [ "$step" = a ]; then
  for r in 1 2; do cb $OLD old text384 100000 _r$r; cb $NEW new text384 100000 _r$r; done
fi
if [ "$step" = all ] || [ "$step" = b ]; then
  for ds in glove sift; do cb $OLD old $ds 100000 ""; cb $NEW new $ds 100000 ""; done
fi
if [ "$step" = all ] || [ "$step" = c ]; then
  eb $OLD old 100000 scaling,loop ""; eb $NEWEB new 100000 scaling,loop,batch ""
fi
if [ "$step" = all ] || [ "$step" = d ]; then
  eb $OLD old 1000000 mem ""; eb $NEWEB new 1000000 mem ""
  for r in 1 2; do eb $OLD old 1000000 writes _r$r; eb $NEWEB new 1000000 writes _r$r; done
  eb $OLD old 1000000 deletes ""; eb $NEWEB new 1000000 deletes ""
fi
if [ "$step" = all ] || [ "$step" = e ]; then
  cb $OLD old text384 1000000 ""; cb $NEW new text384 1000000 ""
fi
echo DONE
