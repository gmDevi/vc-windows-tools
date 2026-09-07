#!/usr/bin/env bash
# Robust parallel ranged download with per-part size verification and resume: pget2.sh <url> <out> [parts]
URL="$1"; OUT="$2"; P="${3:-12}"
SIZE=$(curl -sIL "$URL" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
[ -n "$SIZE" ] || { echo "no size"; exit 1; }
CH=$(( (SIZE + P - 1) / P ))
fetch_part() {
  i=$1; S=$((i*CH)); E=$(( (i+1)*CH - 1 )); [ $E -ge $SIZE ] && E=$((SIZE-1)); want=$((E-S+1)); f="$OUT.part$i"
  for attempt in $(seq 1 30); do
    have=0; [ -f "$f" ] && have=$(stat -c %s "$f")
    [ "$have" -ge "$want" ] && { [ "$have" -gt "$want" ] && truncate -s "$want" "$f"; return 0; }
    curl -s -L --retry 3 -r "$((S+have))-$E" -o - "$URL" >> "$f" || sleep 3
  done
  return 1
}
for i in $(seq 0 $((P-1))); do fetch_part $i & done
wait
ok=1; for i in $(seq 0 $((P-1))); do S=$((i*CH)); E=$(( (i+1)*CH - 1 )); [ $E -ge $SIZE ] && E=$((SIZE-1)); want=$((E-S+1)); have=$(stat -c %s "$OUT.part$i" 2>/dev/null || echo 0); [ "$have" -eq "$want" ] || { echo "part $i incomplete: $have/$want"; ok=0; }; done
[ $ok -eq 1 ] || { echo "INCOMPLETE"; exit 1; }
cat $(for i in $(seq 0 $((P-1))); do echo "$OUT.part$i"; done) > "$OUT" && rm -f "$OUT".part*
echo "size $(stat -c %s "$OUT") expected $SIZE"
