import csv
import json
import os
import sys

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

N = 8
csv.field_size_limit(10**9)
model, tok = load("models/mlx-qwen3-8b-4bit")
rows = [r for f in ("ds/aci/valid.csv", "ds/aci/clinicalnlp_taskB_test1.csv") for r in csv.DictReader(open(f))]
rows = [r for r in rows if len(tok.encode(r["dialogue"])) < 3000][: 2 * N]

SYSTEM = "You are an autonomous clinical documentation agent. You write and maintain the patient's clinical note in the EHR. Keep the note's section headings, keep every detail that is still true, and write in the same style as the existing chart."


def chat(user, system=SYSTEM):
    return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


cache = "ds/aci/followups.jsonl"
done = {r["encounter_id"]: r["followup"] for r in map(json.loads, open(cache))} if os.path.exists(cache) else {}
with open(cache, "a") as fh:
    for r in rows[N:]:
        if r["encounter_id"] in done:
            continue
        p = chat(f"Here is a clinical note from a patient's last visit.\n\n{r['note']}\n\nWrite the transcript of this patient's follow-up visit six weeks later, in the same format as an ambient recording: alternating lines starting with [doctor] and [patient], lowercase, about 40 lines. Some problems have improved, one medication dose changes, a new minor complaint comes up, and the doctor examines the patient and states the results out loud. Output only the transcript.", system="You write realistic synthetic doctor-patient conversations for testing clinical software.")
        done[r["encounter_id"]] = generate(model, tok, p, max_tokens=1200, sampler=make_sampler(0.0))
        fh.write(json.dumps({"encounter_id": r["encounter_id"], "followup": done[r["encounter_id"]]}) + "\n")
        fh.flush()
        print("followup", r["encounter_id"], file=sys.stderr)

with open("data/clinical_workload.jsonl", "w") as fh:
    for i, r in enumerate(rows[:N]):
        fh.write(json.dumps({"id": f"new_visit-{i:02d}", "kind": "new_visit", "encounter_id": r["encounter_id"], "prompt": chat(f"Transcript of today's visit:\n\n{r['dialogue']}\n\nWrite the clinical note for this visit with these sections: CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, REVIEW OF SYSTEMS, PHYSICAL EXAMINATION, RESULTS, ASSESSMENT AND PLAN. Output only the note.")}) + "\n")
    for i, r in enumerate(rows[N:]):
        fh.write(json.dumps({"id": f"follow_up-{i:02d}", "kind": "follow_up", "encounter_id": r["encounter_id"], "prompt": chat(f"The patient is back for a follow-up visit. Here is the current note in their chart:\n\n{r['note']}\n\nTranscript of today's follow-up visit:\n\n{done[r['encounter_id']]}\n\nUpdate the note for today's visit. Carry forward everything that is still accurate, revise what changed, and add the new findings. Output the complete updated note only.")}) + "\n")
print("wrote data/clinical_workload.jsonl", 2 * N, "prompts")
